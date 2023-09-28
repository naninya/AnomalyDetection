import os
import tqdm
import numpy as np
from itertools import chain
# torch
import torch
import torch.nn.functional as F
from sklearn import metrics
from .feature_extractors import PatchFeatureExtractor
from .faiss_nn import FaissNN
from .metrics import compute_pixelwise_retrieval_metrics


class PatchCoreModel(torch.nn.Module):
    def __init__(self, device, backbone_name, flatten_dimension, out_dimension):
        super(PatchCoreModel, self).__init__()
        self.device = device
        self.backbone_name = backbone_name
        self.flatten_dimension = flatten_dimension
        self.out_dimension = out_dimension
        self.feature_extractor = PatchFeatureExtractor(
            device=device, 
            backbone_name=backbone_name, 
            flatten_dimension=flatten_dimension, 
            out_dimension=out_dimension
        )
        on_gpu = device == "cuda"
        self.faiss_nn = FaissNN(on_gpu=on_gpu, num_workers=8)
        self.anomaly_score_th = None
        self.eval()
        self.to(device)
        
    def forward(self, batch_images):
        with torch.no_grad():
            torch.cuda.empty_cache()
            batch_images = batch_images.to("cuda").to(torch.float32)
            batchsize = batch_images.shape[0]
            features, scales = self.feature_extractor(batch_images)
            features = np.asarray(features.detach().cpu().numpy())
            query_distances, query_nns = self.faiss_nn.run(1, features)
            query_distances = np.mean(query_distances, axis=-1)

            # unpatch : check for image score 
            patch_scores = image_scores = query_distances.reshape(batchsize, -1, *query_distances.shape[1:]).copy()
            image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
            image_scores = image_scores.max(axis=1).flatten()
            
            # for check patch image
            patch_scores = patch_scores.reshape(batchsize, scales[0], scales[1])
            patch_scores = torch.from_numpy(patch_scores)
            _scores = patch_scores.to("cuda")
            _scores = _scores.unsqueeze(1)
            _scores = F.interpolate(
                _scores, size=batch_images.shape[2], mode="bilinear", align_corners=False
            )
            _scores = _scores.squeeze(1)
            patch_scores = _scores.cpu().numpy() 
        return image_scores, patch_scores
        
    def fit(self, dataloader, update_th=True):
        # Extracting features with patch format
        with torch.no_grad():
            torch.cuda.empty_cache()
            out_features = []
            for batch in tqdm.tqdm(dataloader):
                batch_image_paths = batch["image_path"]
                batch_images = batch["image"].to(torch.float32)
                batch_labels = batch["label"]
                
                batch_images = batch_images.to("cuda")
                batch_labels = batch_labels.to("cuda")
                out, ref_num_patches =self.feature_extractor(batch_images)
                out = out.detach().cpu().numpy() 
                out_features.append(out)
            features = np.concatenate(out_features, axis=0)
            print(f"extracted all feature shape: {features.shape}")
            # Sampling features
            features = self.feature_extractor.sampler.run(features)
            print(f"sampled feature shape: {features.shape}")
            # train features
            self.faiss_nn.fit(features)
            
            
        #     self.update_threshold(th=None, train_dataloader=dataloader)

    def inference(self, dataloader):
        pred_image_scores = []
        pred_masks = []
        annotations = []
        anomaly_labels = []
        image_paths = []
        for i, batch in tqdm.tqdm(enumerate(dataloader)):
            batch_image_paths = batch["image_path"]
            inputs = batch["image"]
            labels = batch["label"]

            inputs = inputs.to(self.device).to(torch.float)
            # forward
            image_score, patch_score = self.forward(inputs)
            # images scores
            pred_image_scores.append(image_score)
            # pred masks
            pred_masks.append(patch_score)
            # labels
            anomaly_labels.append(labels)
            # image paths
            image_paths.append(batch_image_paths)
            # annotations
            annotations.append(batch["mask"])
        pred_image_scores = np.concatenate(pred_image_scores)
        anomaly_labels = np.concatenate(anomaly_labels)
        pred_masks = np.concatenate(pred_masks, axis=0)
        image_paths = np.array(list(chain(*image_paths)))
        annotations = np.concatenate(annotations)

        _, _, thresholds = metrics.roc_curve(
            anomaly_labels, pred_image_scores
        )
        accuracies = []
        for th in thresholds:
            pred = pred_image_scores >= th
            accuracies.append((pred == anomaly_labels).mean())
        
        self.anomaly_score_th = thresholds[np.array(accuracies).argmax()]
        accuracy = np.array(accuracies).max()
        pred_labels = pred_image_scores >= self.anomaly_score_th
        auroc = compute_pixelwise_retrieval_metrics(pred_masks, annotations)["auroc"]
        return {
            "anomaly_scores":pred_image_scores,
            "anomaly_labels":anomaly_labels.astype(np.bool_),
            "pred_masks":pred_masks,
            "image_paths":image_paths,
            "pred_labels":pred_labels,
            "accuracy":accuracy,
            "annotations":annotations,
            "auroc":auroc,
            "th":self.anomaly_score_th
        }
    
    # def update_threshold(self, th, train_dataloader=None):
    #     if th is not None:
    #         return self.anomaly_score_th 
    #     self.anomaly_score_th = self.inference(train_dataloader)["anomaly_scores"].max()
    #     return self.anomaly_score_th
    
    def save(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        self.faiss_nn.save(os.path.join(save_dir, "faiss_weight"))
        
    def load(self, save_dir, device, backbone_name, flatten_dimension, out_dimension):
        self.__init__(device, backbone_name, flatten_dimension, out_dimension)
        self.faiss_nn.load(os.path.join(save_dir, "faiss_weight"))
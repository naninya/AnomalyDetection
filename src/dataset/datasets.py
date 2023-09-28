import os
import torch
import cv2
import numpy as np
from glob import glob
from PIL import Image

class AnomalyDataset(torch.utils.data.Dataset): 
    def __init__(
        self,
        root_dir,
        step,
        transform=None
    ):
        self.root_dir = root_dir
        self.step = step
        self.transform = transform
        self.image_paths = []
        self.annotation_paths = []
        self.labels = []
        self._split()
    def __len__(self):
        return len(self.image_paths)
    
    def _split(self):
        # file name should contain anomaly or normal
        assert len(sorted(glob(f"{self.root_dir}/images/*.*"))) > 0
        for index, image_path in enumerate(sorted(glob(f"{self.root_dir}/images/*.*"))):
            image_id = os.path.basename(image_path).split(".")[0]
            for annotation_path in sorted(glob(f"{self.root_dir}/annotations/*.*")):
                if image_id == os.path.basename(annotation_path).split(".")[0]:
                    break
            else:
                annotation_path = None
                
            # anomaly
            label = ("anomaly" in image_path) * 1.0
            self.image_paths.append(image_path)
            self.annotation_paths.append(annotation_path)
            self.labels.append(label)
            

    def __getitem__(self, idx): 
        image_path = self.image_paths[idx]
        annotation_path = self.annotation_paths[idx]
        image = np.array(Image.open(image_path).convert("RGB"))/255
        if annotation_path is not None:
            annotation = np.array(Image.open(annotation_path))[:,:]
            annotation = np.where(annotation > 0, 1, 0)
        else:
            annotation = np.zeros(image.shape[:2])
        label = self.labels[idx]
        sample = dict(
            image=image,
            mask=annotation,
            image_path=image_path,
            label=label,
        )

        if self.transform is not None:
            sample = self.transform(**sample)
            return sample
        return sample

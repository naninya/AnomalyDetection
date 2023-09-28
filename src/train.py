import os
import pickle
from time import time
# import custom modules
from utils import seed_everything_custom, show_result_image
from dataset.utils import reform_mvtec_dir_tree, get_dataset, get_dataloader
from dataset.transformers import Transformers
from dataset.datasets import AnomalyDataset
from model.models import PatchCoreModel


def train():
    seed_everything_custom()
    # hyperparameters
    configs = {}
    for key in [
        "bottle",
        "cable",
        "capsule",
        "carpet",
        "hazelnut",
        "metal_nut",
        "pill"
    ]:
        configs[key] = {
                "image_size" : 256,
                "batch_size" : 2,
                "backbone":"wideresnet50",
                "device" : "cuda",
                "train_data_dir" : f"../data/{key}/train",
                "valid_data_dir" : f"../data/{key}/test",
                "model_weight_dir" : "../result",
        }

    results = {}
    for key, config in configs.items():
        start = time()
        # trainsformer
        transforms = Transformers(config["image_size"])
        # dataset
        train_dataset = get_dataset(config["train_data_dir"], "train", transforms.train_transformer)
        train_dataloader = get_dataloader(train_dataset, config["batch_size"], shuffle=False, drop_last=False)
        valid_dataset = get_dataset(config["valid_data_dir"], "valid", transforms.valid_transformer)
        valid_dataloader = get_dataloader(valid_dataset, config["batch_size"], shuffle=False, drop_last=False)
        
        best_accuracy = 0
        sub_result = {}
        for model_index, dimension in enumerate([512, 1024, 2048, 4096]):
            print(f"""\n
            ====target:{key}::dimension:{dimension}====
            \n
            """)
            model_path = os.path.join(config["model_weight_dir"], f"{key}_{dimension}")
            
            # model
            model = PatchCoreModel(
                device = config["device"],
                backbone_name = config["backbone"],
                flatten_dimension = dimension,
                out_dimension = dimension
            )
            if not os.path.isdir(model_path):
                # train
                model.fit(train_dataloader, update_th=False)
            else:
                model.load(        
                    save_dir = model_path,
                    device = config["device"],
                    backbone_name = config["backbone"],
                    flatten_dimension = dimension,
                    out_dimension = dimension
                )
            # test
            test_result = model.inference(valid_dataloader)
            acc = test_result["accuracy"]
            print(f"dimension{dimension}'s accuracy:{acc}'")
            if acc > best_accuracy:
                best_accuracy = acc
                sub_result[model_index] = test_result
                model.save(model_path)
            else:
                break
            del model
        
        best_model_result = sorted(sub_result.items(), key=lambda x:x[1]["accuracy"], reverse=True)[0][1]
        # save 
        acc = best_model_result["accuracy"]
        auroc = best_model_result["auroc"]
        best_model_result["time"] = time() - start
        print(f"TEST ACCURACY:{acc}, TEST AUROC:{auroc}")
        results[key] = best_model_result
    result_dir = config["model_weight_dir"]
    with open(f"{result_dir}/result.pkl", "wb") as f:
        pickle.dump(results, f)

if __name__ == "__main__":
    train()
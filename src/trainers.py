import os
import pickle
from time import time
# import custom modules
from utils import seed_everything_custom, show_result_image
from dataset.utils import reform_mvtec_dir_tree, get_dataset, get_dataloader
from dataset.transformers import Transformers
from dataset.datasets import AnomalyDataset
from model.models import PatchCoreModel


class Trainer:
    def __init__(self, configs):
        self.configs = configs
        # setup
        for name in configs.TARGET_DATASET_NAMES:
            os.makedirs(os.path.join(configs.MODEL_ROOT, name), exist_ok=True)
            os.makedirs(os.path.join(configs.RESULT_ROOT, name), exist_ok=True)
    def run(self):
        seed_everything_custom()
        # hyperparameters
        key_configs = {}
        for key in self.configs.TARGET_DATASET_NAMES:
            key_configs[key] = {
                "image_size" : self.configs.IMAGE_SIZE,
                "batch_size" : self.configs.BATCH_SIZE,
                "backbone": self.configs.BACKBONE_NAME,
                "device" : self.configs.DEVICE,
                "train_data_dir" : f"{self.configs.DATA_ROOT}/{key}/train",
                "valid_data_dir" : f"{self.configs.DATA_ROOT}/{key}/valid",
                "model_dir" : os.path.join(self.configs.MODEL_ROOT, key),
                "result_dir" : os.path.join(self.configs.RESULT_ROOT, key),
            }

        for key, config in key_configs.items():
            start = time()
            # trainsformer
            transforms = Transformers(config["image_size"])
            # dataset
            train_dataset = get_dataset(config["train_data_dir"], transforms.train_transformer)
            train_dataloader = get_dataloader(train_dataset, config["batch_size"], shuffle=False, drop_last=False)
            valid_dataset = get_dataset(config["valid_data_dir"], transforms.valid_transformer)
            valid_dataloader = get_dataloader(valid_dataset, config["batch_size"], shuffle=False, drop_last=False)

            best_accuracy = 0
            sub_result = {}
            for model_index, dimension in enumerate([512, 1024, 2048, 4096]):
                print(f"""\n
                ====target:{key}::dimension:{dimension}====
                \n
                """)

                # model
                model = PatchCoreModel(
                    device = config["device"],
                    backbone_name = config["backbone"],
                    flatten_dimension = dimension,
                    out_dimension = dimension
                )
                model.fit(train_dataloader)
                valid_result = model.inference(valid_dataloader)
                acc = valid_result["accuracy"]
                print(f"dimension{dimension}'s accuracy:{acc}'")
                if acc > best_accuracy:
                    best_accuracy = acc
                    sub_result[model_index] = valid_result
                    save_path = os.path.join(config["model_dir"], str(dimension))
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    model.save(save_path)
                else:
                    break
        #             pass
                del model

            best_model_result = sorted(sub_result.items(), key=lambda x:x[1]["accuracy"], reverse=True)[0][1]
            # save 
            acc = best_model_result["accuracy"]
            auroc = best_model_result["auroc"]
            best_model_result["time"] = time() - start
            print(f"TEST ACCURACY:{acc}, TEST AUROC:{auroc}")
            with open(os.path.join(config["result_dir"], "result.pkl"), "wb") as f:
                pickle.dump(best_model_result, f)

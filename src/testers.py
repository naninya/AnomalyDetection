import os
import pickle
from time import time
# import custom modules
from utils import seed_everything_custom, show_result_image
from dataset.utils import reform_mvtec_dir_tree, get_dataset, get_dataloader
from dataset.transformers import Transformers
from dataset.datasets import AnomalyDataset
from model.models import PatchCoreModel


class Tester:
    def __init__(self, configs):
        self.configs = configs
    def run(self, model_path, target_path, output_path):
        seed_everything_custom()

        start = time()
        # trainsformer
        transforms = Transformers(self.configs.IMAGE_SIZE)
        # dataset
        test_dataset = get_dataset(target_path, transforms.train_transformer)
        test_dataloader = get_dataloader(test_dataset, self.configs.BATCH_SIZE, shuffle=False, drop_last=False)
        for model_index, dimension in enumerate([512, 1024, 2048, 4096][::-1]):
            target_model_path = os.path.join(model_path, str(dimension))
            if not os.path.isdir(target_model_path):
                continue
            model = PatchCoreModel(
                device = self.configs.DEVICE,
                backbone_name = self.configs.BACKBONE_NAME,
                flatten_dimension = dimension,
                out_dimension = dimension
            )
            model.load(
                save_dir=target_model_path,
                device = self.configs.DEVICE,
                backbone_name = self.configs.BACKBONE_NAME,
                flatten_dimension = dimension,
                out_dimension = dimension
            )
            test_result = model.inference(test_dataloader)
            break
    
        test_result["time"] = time() - start
        with open(os.path.join(output_path, "test_result.pkl"), "wb") as f:
            pickle.dump(test_result, f)

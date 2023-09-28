import os
import re
import shutil
from glob import glob
from torch.utils.data import DataLoader
from .datasets import AnomalyDataset
from .transformers import Transformers

def load_transform(image_size):
    transform = Transformers(image_size)
    return transform

def get_dataset(data_dir:str, setp, transform):
    dataset = AnomalyDataset(
        root_dir=data_dir,
        step=setp,
        transform=transform
    )
    return dataset

def get_dataloader(dataset, batch_size, shuffle=True, drop_last=True):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    return dataloader
    

def reform_mvtec_dir_tree(
        source_dir="../notebook/sample_data", 
        target_dir="../data"
    ):
    for _source in glob(f"{source_dir}/*"):
        category = os.path.basename(_source)

        reg = re.compile(r"\/(?P<file_num>\d{3})")
        def get_file_id(path):
            split_data = path.split("/")
            type_id= split_data[-2]
            file_num = reg.search(path).group("file_num")
            return f"{type_id}_{file_num}"

        for index, image_path in enumerate(
                sorted(glob(f"{source_dir}/{category}/train/good/*.*"))
            ):
            des_path = os.path.join(target_dir, f"{category}", "train", "images", f"normal_{os.path.basename(image_path)}")
            os.makedirs(os.path.dirname(des_path), exist_ok=True)
            shutil.copy(image_path, des_path)
            
        for index, image_path in enumerate(sorted(glob(f"{source_dir}/{category}/test/**/*.*"))):
            if "good" in image_path:
                des_path = os.path.join(target_dir, f"{category}", "test", "images", f"normal_{os.path.basename(image_path)}")
            else:
                error_name = image_path.split("/")[-2]
                des_path = os.path.join(target_dir, f"{category}", "test", "images", f"anomaly_{error_name}_{os.path.basename(image_path)}")
            os.makedirs(os.path.dirname(des_path), exist_ok=True)
            shutil.copy(image_path, des_path)
            
        for index, image_path in enumerate(sorted(glob(f"{source_dir}/{category}/ground_truth/**/*.*"))):
            error_name = image_path.split("/")[-2]
            _name = os.path.basename(image_path).split("_mask")[0]
            mask_name = f"{_name}.png"
            
            des_path = os.path.join(target_dir, f"{category}", "test", "annotations", f"anomaly_{error_name}_{mask_name}")
            os.makedirs(os.path.dirname(des_path), exist_ok=True)
            shutil.copy(image_path, des_path)
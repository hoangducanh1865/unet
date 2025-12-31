import os
import torch
import random
import numpy as np
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ADE20KDataset(Dataset):
    def __init__(
        self, path_to_data, image_size=128, train=True, inference=False, random_crop_ratio=(0.08, 1)
    ):
        # self.path_to_data = path_to_data
        # self.inference = inference
        # self.train = train
        # self.image_size = image_size
        self.min_ratio, self.max_ratio = random_crop_ratio
        self.split = "training" if train else "validation"

        self.path_to_images = os.path.join(path_to_data, "images", self.split)
        self.path_to_annotations = os.path.join(path_to_data, "annotaions", self.split)
        self.file_roots = [path.split(".")[0] for path in os.listdir(self.path_to_images)]

        import inspect

        _frame = inspect.currentframe()
        print(
            f'Start debug in file "/Volumes/MTS800/tmp/it/ai_engineer_learning_path/priyam_mazumdar_tutorials/unet/dataloader.py", line {_frame.f_lineno}'
        )
        # Your debug code here
        print(self.file_roots)
        print(
            f'End debug in file "/Volumes/MTS800/tmp/it/ai_engineer_learning_path/priyam_mazumdar_tutorials/unet/dataloader.py", line {_frame.f_lineno}'
        )

        self.resize = transforms.Resize((image_size, image_size))
        self.normalize = transforms.Normalize(
            mean=(0.48897059, 0.46548275, 0.4294), std=(0.22861765, 0.22948039, 0.24054667)
        )
        self.random_resize=transforms.RandomResizedCrop(size=(image_size, image_size),
                                                        scale=(self.min_ratio, self.max_ratio))

def test():
    path_to_data = "data/ADE20K"
    dataset = ADE20KDataset(path_to_data=path_to_data)


if __name__ == "__main__":
    test()

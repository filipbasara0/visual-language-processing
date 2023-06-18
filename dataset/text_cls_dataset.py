import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
import PIL
import random
import numpy as np


class VLPTextClassificationDataset(Dataset):

    def __init__(self, data, training=False):
        self.data = data
        self.training = training

    def __getitem__(self, index):
        image_path, target = self.data[index]
        image = PIL.Image.open(image_path)

        if self.training and random.random() < 0.5:
            image = PIL.Image.fromarray(np.invert(image))

        image = F.to_tensor(image)

        target = torch.tensor(target)
        return image, target

    def __len__(self):
        return len(self.data)
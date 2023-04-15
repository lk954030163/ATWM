# 日期:2022年5月7日
# 时间：16:26
# 自定义数据集
"""
对抗样本数据并行加载
"""
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os, glob, random, csv
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from torchvision.utils import make_grid
import numpy as np


class AdvDataset(Dataset):
    def __init__(self, list, resize):
        """

        list：
        """
        super(AdvDataset, self).__init__()

        self.list = list
        self.resize = resize

    def __len__(self):
        return len(self.list)

    def __getitem__(self, idx):

        img = self.list[idx]
        label = torch.tensor(1)

        return img, label

# -*- coding: utf-8 -*-
# 2021.10.25
# 自定义数据集
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os, glob, random, csv
import torch, math
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from torchvision.utils import make_grid
import numpy as np
from PIL import Image, ImageFile
from utils.byte2img import create_file

# ImageFile.LOAD_TRUNCATED_IMAGES = True
"""
"""


# load_csv('./data/train','train.csv')
# load_csv('./data/validation','val.csv')
# load_csv('./data/test','test.csv')


class MyDataset(Dataset):
    def __init__(self, root, resize, mode, imgclass):
        """

        :param root:存放数据的目录
        :param resize: 图片重整大小
        :param mode: 设定该数据集用途（训练）（评估）（测试）
        """
        super(Dataset, self).__init__()

        self.root = root
        self.resize = resize
        self.name = str(mode) + ".csv"
        self.imgclass = imgclass

        self.images = []
        self.labels = []
        # 文件存在，打开获取每一行的内容
        with open(os.path.join(self.root, self.name)) as f:
            reader = csv.reader(f)
            for row in reader:
                img, label = row
                label = int(label)

                self.images.append(img)
                self.labels.append(label)
        assert len(self.images) == len(self.labels)

    def __len__(self):
        return len(self.images)

    def img2img(self, file_path):
        img_in = Image.open(file_path).convert("L")
        img_list = np.array(img_in)
        c, w = img_list.shape
        new_list = []
        for i in range(0, c):
            temp_list = img_list[i][:]
            list_var = np.var(temp_list)
            if list_var > 4:
                new_list.append(temp_list)

        binary_lsit = [j for i in new_list for j in i]
        size = int(math.sqrt(len(binary_lsit)))
        img_out = create_file(binary_lsit, (size + 1, size + 1), "L")

        return img_out

    def __getitem__(self, idx):
        # img:\root\name\*.png
        img_name, label = self.images[idx], self.labels[idx]

        if self.imgclass == "virusdata":
            if img_name.find("adv_data") == -1:
                img_name = img_name.replace("malware", "new_malware")
        if self.imgclass == "big2015":
            if img_name.find("adv_data") == -1:
                img_name = img_name.replace("bigimg2015", "new_bigimg2015")
        if self.imgclass == "realdataimg":
            if img_name.find("adv_data") == -1:
                img_name = img_name.replace(
                    "realdataimg", "realdataimg/new_realdataimg"
                )

        transform = transforms.Compose(
            [
                lambda x: Image.open(x).convert("L"),  # 从路径中获取图片
                # lambda x: self.img2img(x),
                # transforms.RandomRotation(53), #随机旋转
                # transforms.CenterCrop(self.resize), #中心旋转
                transforms.ToTensor(),  # 转张量
                transforms.Resize((self.resize, self.resize)),  # 转换图片大小
                # transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.255]) #正态化
            ]
        )

        """img=Binary2img(img_name)
        img_name=img_name.replace("data","imgdata")
        if img_name.find(".")!=-1:
            img_name=img_name.replace(".","b")
            img_name=img_name.replace("b/","./")
        img=img.createGreyScaleImage()
        img.save(img_name+".jpg")"""

        # 测试点
        img = transform(img_name)
        # img.save("img_name.jpg")
        label = torch.tensor(label)
        return img, label

    def load_csv(path, rate):
        """
        建立数据集索引
        划分测试训练集
        未作代码复用
        未添加进类
        """
        # 首先判断是否存在，如果不存在则从数据集中生成CSV文件
        filename1 = "train.csv"
        filename2 = "test.csv"
        if os.path.exists(os.path.join(path, filename1)):
            print("train.CSV文件已存在")
            if os.path.exists(os.path.join(path, filename2)):
                print("test.CSV文件已存在")

        else:
            name2label = {}  # 编码映射表

            # 对root路径下的文件夹进行遍历
            for name in sorted(os.listdir(os.path.join(path))):
                if not os.path.isdir(os.path.join(path, name)):
                    continue
                name2label[name] = len(name2label.keys())

            files = []
            # 遍历映射中的name，根据命名规则/root/name/*.文件格式
            for name in name2label.keys():
                # 符合\root\name\*  的文件映射进files里
                names = []
                names += glob.glob(os.path.join(path, name, "*"))
                # images += glob.glob(os.path.join(path, name, '*.jpeg'))
                files.append(names)

            # 根据比例划分测试，训练集
            trains = []
            tests = []
            for i in range(0, len(files)):
                temp = files[i]
                random.shuffle(temp)
                offset = int(len(temp) * rate)
                if len(files) == 0 or offset < 1:
                    print("ERROR")
                    break
                train = temp[:offset]
                test = temp[offset:]
                trains = trains + train
                tests = tests + test

            # 打乱顺序
            random.shuffle(trains)
            random.shuffle(tests)

            # 新建指定的CSV文件
            with open(os.path.join(path, filename1), mode="w", newline="") as f:
                writer = csv.writer(f)
                for img in trains:
                    # 分割 \root\name\*
                    name = img.split(os.sep)[-2]
                    label = name2label[name]
                    # if label==0:
                    # continue
                    # 行写入\root\name\*
                    writer.writerow([img, label])
                print("写入train.CSV文件  完成！")

            with open(os.path.join(path, filename2), mode="w", newline="") as f:
                writer = csv.writer(f)
                for img in tests:
                    # 分割 \root\name\*
                    name = img.split(os.sep)[-2]
                    label = name2label[name]
                    # 行写入\root\name\*
                    writer.writerow([img, label])
                print("写入test.CSV文件  完成！")


# MyDataset.load_csv("./malimg/val",1)
"""
#测试部分
MyDataset.load_csv("./data",0.7)
transformed_trainset = MyDataset('./data', 512, mode='train')
transformed_trainset.__getitem__(0)
trainset_dataloader = DataLoader(dataset=transformed_trainset,
                                 batch_size=4,
                                 shuffle=True,
                                 num_workers=0)

"""

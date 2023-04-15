# data to adv_data
#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"  #
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
import sys, os, time
from Log import Logger
import torchvision.models as models
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.AdvGenerateData import MyDataset
import torch
from torchvision import utils as vutils
import random, csv, math
import random
from queue import Queue
from threading import Thread
from tqdm import tqdm
from multiprocessing import Process
from utils.attackers import FGSM, BIM
import itertools
import yaml
from utils.AddCsv import add_csv
from itertools import product


def write2csv(csvdata, filename):
    random.shuffle(csvdata)
    with open(filename, mode="w", newline="") as f:
        writer = csv.writer(f)
        for img in csvdata:
            writer.writerow(img)
        print("写入CSV文件  完成！")


def save_image_tensor(input_tensor: torch.Tensor, filename):
    """
    tensor2img
    :param input_tensor
    :param filename
    """
    # assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 64)
    # copy  AND To cpu
    input_tensor = input_tensor.clone().detach()
    input_tensor = input_tensor.to(torch.device("cpu"))
    # input_tensor = unnormalize(input_tensor)   # unnormalize
    vutils.save_image(input_tensor, filename)


def injectimg(x, s, indexs, adv_xx):
    x = x.cpu()
    c, h, w = x.shape

    resize = transforms.Resize([w, w])
    adv_xx = resize(adv_xx)
    adv_xx = adv_xx.cpu()

    original_x = x.clone()
    original_x = original_x.squeeze(0).unsqueeze(2)
    original_x = original_x.detach().numpy()

    x = x.view(-1).detach().numpy().tolist()
    adv_xx = adv_xx.view(-1).detach().numpy().tolist()

    catlist = []
    for i in range(0, 4):
        catimg = x[:]
        catlist.append(catimg)

    try:
        indexs = list(indexs)
    except:
        indexs = [indexs]

    indexs.sort(reverse=True)
    indexs = [int(j * 0.01 * w) for j in indexs]
    s = int(s * 0.01 * w)
    for i in range(0, len(indexs)):

        attack_flag = 0
        idx = indexs[i] * w
        if attack_flag == 0:
            random_temp = [random.randint(0, 255) for i in range(s * w)]
            cat = [x / 255 for x in random_temp]
            catimg = catlist[0]
            catimg[idx:idx] = cat
            catlist[0] = catimg[:]
        if 0:
            if s % 4 == 0:
                attack_flag = 1
            else:
                attack_flag = 2
            s_falg = 0.04
            idx = indexs[i] * w
            if attack_flag == 1:
                if s < s_falg * w:
                    random_temp = [random.randint(0, 0) for i in range(s * w)]
                    cat = random_temp
                    catimg = catlist[1]
                    catimg[idx:idx] = cat
                    catlist[1] = catimg[:]

            idx = indexs[i] * w
            if attack_flag == 2:
                if s < s_falg * w:
                    random_temp = [random.randint(1, 1) for i in range(s * w)]
                    cat = random_temp
                    catimg = catlist[2]
                    catimg[idx:idx] = cat
                    catlist[2] = catimg[:]

        attack_flag = 3
        idx = indexs[i] * w
        if attack_flag == 3:
            cat = adv_xx[idx : s * w + idx]
            catimg = catlist[3]
            catimg[idx:idx] = cat
            catlist[3] = catimg[:]

    catimglist = []
    for i in range(0, len(catlist)):
        cat_is = catlist[i]
        m = int(math.sqrt(len(cat_is)))
        m = m + 1 if m * m != len(cat_is) else m
        for j in range(0, m * m - len(cat_is)):
            cat_is.append(0)  # padding 0

        cat_is = torch.Tensor(cat_is)
        catimg = cat_is.view(m, m)
        catimg = catimg.unsqueeze(0)
        catimglist.append(catimg)

    return catimglist


def replaceimg(x, s, indexs, adv_xx):
    x = x.cpu()
    c, h, w = x.shape

    resize = transforms.Resize([w, w])
    adv_xx = resize(adv_xx)
    adv_xx = adv_xx.cpu()

    original_x = x.clone()
    original_x = original_x.squeeze(0).unsqueeze(2)
    original_x = original_x.detach().numpy()

    x = x.view(-1).detach().numpy().tolist()
    adv_xx = adv_xx.view(-1).detach().numpy().tolist()

    replacelist = []
    for i in range(0, 4):
        replaceimg = x[:]
        replacelist.append(replaceimg)
    try:
        indexs = list(indexs)
    except:
        indexs = [indexs]
    indexs.sort(reverse=True)
    indexs = [int(j * 0.01 * w) for j in indexs]
    s = int(s * 0.01 * w)
    for i in range(0, len(indexs)):

        attack_flag = 0
        idx = indexs[i] * w
        if attack_flag == 0:
            random_temp = [random.randint(0, 255) for i in range(s * w)]
            replace = [x / 255 for x in random_temp]
            replaceimg = replacelist[0]
            replaceimg[idx : idx + s * w] = replace
            replacelist[0] = replaceimg[:]

        if 0:
            if s % 4 == 0:
                attack_flag = 1
            else:
                attack_flag = 2

            s_falg = 0.04
            idx = indexs[i] * w
            if attack_flag == 1:
                if s < s_falg * w:
                    random_temp = [random.randint(0, 0) for i in range(s * w)]
                    replace = random_temp
                    replaceimg = replacelist[1]
                    replaceimg[idx : idx + s * w] = replace
                    replacelist[1] = replaceimg[:]

            idx = indexs[i] * w
            if attack_flag == 2:
                if s < s_falg * w:
                    random_temp = [random.randint(1, 1) for i in range(s * w)]
                    replace = random_temp
                    replaceimg = replacelist[2]
                    replaceimg[idx : idx + s * w] = replace
                    replacelist[2] = replaceimg[:]

        attack_flag = 3
        idx = indexs[i] * w
        if attack_flag == 3:
            replace = adv_xx[idx : s * w + idx]
            replaceimg = replacelist[3]
            replaceimg[idx : idx + s * w] = replace
            replacelist[3] = replaceimg[:]

    replaceimglist = []
    for i in range(0, len(replacelist)):
        replace_is = replacelist[i]
        m = int(math.sqrt(len(replace_is)))
        m = m + 1 if m * m != len(replace_is) else m
        for j in range(0, m * m - len(replace_is)):
            replace_is.append(0)  # padding 0

        replace_is = torch.Tensor(replace_is)
        replaceimg = replace_is.view(m, m)
        replaceimg = replaceimg.unsqueeze(0)
        replaceimglist.append(replaceimg)

    return replaceimglist


def yaml2name(labels, name):
    yamldata = read_yaml("./config/data.yaml")
    source_data = getargv()  # "big2015"  #
    yamldata = yamldata[source_data]
    adv_files = yamldata["adv_files"]
    adv_file_name = adv_files + str(labels) + "/" + name.split("/")[-1]
    return adv_file_name


def generate_mydata(
    xx, labels, s, name, indexs, mode, adv_xx, model, resize_size, counter
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"

    resize = transforms.Resize([resize_size, resize_size])
    # print(name)
    modes = ["cat", "repalce"]
    print("Now is creat:", counter)
    for mode in modes:
        if mode == "cat":
            catimglist = injectimg(xx, s, indexs, adv_xx)
        else:
            catimglist = replaceimg(xx, s, indexs, adv_xx)
        replacename = [
            "random",
            "fgsm",
        ]

        for j in [0, 3]:
            adv_xxj = catimglist[j]
            adv_xxj = resize(adv_xxj)
            adv_xxj = adv_xxj.unsqueeze(0).to(device)
            _, predicted = torch.max(model(adv_xxj).data, 1)

            replacestr = replacename[j]

            if predicted.item() != labels:

                filename = yaml2name(labels, name)
                filename = filename.replace(
                    ".jpg", "_" + mode + "_" + replacestr + "_.jpg"
                )
                # print(filename)
                save_image_tensor(adv_xxj, filename)


def product_indexs():

    loop_length = 2  # big2015

    loop_length = 4
    ride_space = 4
    size = 100
    loop_list = list(range(5, size - 5, int(size / (loop_length + 1))))
    loop_val = []
    for i in range(0, len(loop_list) - 1):
        loop_val.append(list(range(loop_list[i], loop_list[i + 1], ride_space)))

    indexs = []
    for i in product(*loop_val):
        indexs.append(i)

    # indexs = list(range(5, size - 5, 1))

    return indexs


def main(
    data, num_classes, data_path, modelname, mode, thread_number=16,
):
    resize_size = 224
    batchsize = 1

    # model path
    pretrained_model = "./pre_train_model/" + modelname + "_densenet121.mdl"
    print(pretrained_model)
    imgclass = "NO"
    if modelname.find("pre") != -1:
        imgclass = modelname.split("_")[1]
    print(imgclass)
    # GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    model = models.densenet121(pretrained=False)
    model.features.conv0 = nn.Conv2d(
        1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
    )
    for param in model.parameters():
        param.requires_grad = True
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)
    model.load_state_dict(
        torch.load(pretrained_model, map_location=torch.device("cpu")), False
    )
    model.eval()

    # data
    transformed_dataset = MyDataset(data_path, resize_size, data, imgclass)
    dataloader = DataLoader(
        dataset=transformed_dataset, batch_size=batchsize, shuffle=True, num_workers=0
    )
    yamldata = read_yaml("./config/data.yaml")
    source_data = getargv()  # "big2015"  #
    yamldata = yamldata[source_data]

    s1 = int(100 * yamldata["s1"])
    s2 = int(100 * yamldata["s2"])
    ss = list(range(s1, s2, 1))
    fgsm_is = yamldata["fgsm_is"]

    my_indexs = product_indexs()

    counter = 0
    length_load = len(dataloader)
    resize = transforms.Resize([resize_size, resize_size])
    file_queue = Queue()

    for x, y, name in tqdm(dataloader, desc="Now is : "):
        fgsm_flag = counter % fgsm_is + 1  # 3, 2 ,(0,2),i+1
        attacker = FGSM(eps=fgsm_flag * 25 / 255, clip_max=1, clip_min=0)

        x, y = x.to(device), y.to(device)
        adv_x = x.clone().to(device)
        adv_x = resize(adv_x)
        adv_x = attacker.generate(model, adv_x, y)

        name_list = list(name)
        if len(my_indexs) > length_load:
            idx_indexs = list(
                range(0, len(my_indexs), int(len(my_indexs) / length_load))
            )

            indexs = my_indexs[idx_indexs[counter]]
        else:
            counter_idx = counter % len(my_indexs)
            indexs = my_indexs[counter_idx]

        counter += 1
        # for i in range(0, len(name_list)):
        if True:
            i = 0
            xx = x[i]
            adv_xx = adv_x[i]
            adv_xx = adv_xx.unsqueeze(0)

            labels = y[i].item()
            name = name_list[i]
            s = ss[counter % len(ss)]
            # s = s + 1 if i % 2 == 0 else s
            mode = "cat" if counter % 2 == 0 else "replace"

            sample_data = [xx, labels, s, name, indexs, mode, adv_xx, counter]
            file_queue.put(sample_data)
            # thread_number = length_load
    for i in range(thread_number):
        thread = Thread(target=run, args=(file_queue, model, resize_size))
        thread.daemon = True
        thread.start()
    file_queue.join()


def run(file_queue, model, resize_size):

    while not file_queue.empty():
        xx, labels, s, name, indexs, mode, adv_xx, counter = file_queue.get()
        generate_mydata(
            xx, labels, s, name, indexs, mode, adv_xx, model, resize_size, counter
        )
        file_queue.task_done()


def read_yaml(path):
    with open(path, "r", encoding="utf8") as f:
        return yaml.safe_load(f.read())


def dirs2labels(img_root, data_root):
    dirs_list = os.listdir(img_root)
    dirs_list.sort()
    img_list = []
    classlist = []
    for dir in dirs_list:
        labels = dirs_list.index(dir)
        classlist.append([dir, labels])
        now_dir = os.path.join(img_root, dir)
        imgs = os.listdir(now_dir)
        for img in imgs:
            img_data = os.path.join(now_dir, img)
            img_list.append([img_data, labels])

    write2csv(img_list, data_root + "adv_imgdata.csv")
    write2csv(classlist, data_root + "adv_n.csv")


def clear_data(img_root):
    dirs_list = os.listdir(img_root)
    dirs_list.sort()
    classlist = []
    for dir in dirs_list:
        labels = dirs_list.index(dir)
        classlist.append([dir, labels])
        now_dir = os.path.join(img_root, dir)
        imgs = os.listdir(now_dir)
        for img in imgs:
            img_data = os.path.join(now_dir, img)
            try:
                os.remove(img_data)
            except:
                print("Remove Error")

    print("Clear data  is ok")


def getargv():
    try:
        source_data = str(sys.argv[1])
    except:
        source_data = "realdataimg"

    return source_data


if __name__ == "__main__":

    random.seed(1024)

    yamldata = read_yaml("./config/data.yaml")

    source_data = getargv()  # "big2015"  #
    yamldata = yamldata[source_data]

    num_classes = yamldata["num_classes"]
    data_root = yamldata["data_root"]
    start_idx = yamldata["start_idx"]
    modelname = yamldata["modelname"]

    data_path = data_root + "GenerateAdvLabels"
    img_root = data_root + "adv_data"

    start_time = time.time()
    clear_data(img_root)
    """for mode in ["cat", "replace"]:  # ,"replace"
        print("Now is mode:", mode)"""
    mode = "cat+replace"
    for i in range(start_idx, num_classes):
        print("====", mode, "======Now is run data ", i, "=====")
        data = "data" + str(i)
        main(
            data, num_classes, data_path, modelname, mode, thread_number=64,
        )

    dirs2labels(img_root, data_root)
    trains = data_root + "train.csv"
    adv_data = data_root + "adv_imgdata.csv"
    add_csv(trains, adv_data)
    end_time = time.time()
    print("The cost Time: ", end_time - start_time)

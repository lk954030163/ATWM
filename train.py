# 2022/11/30
# model train
import os, sys

try:
    devices = str(sys.argv[4])
    os.environ["CUDA_VISIBLE_DEVICES"] = devices
    if str(sys.argv[4]) == "more":
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
    if str(sys.argv[4]) != "more":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(sys.argv[4])
except:
    devices = "2"
    os.environ["CUDA_VISIBLE_DEVICES"] = devices  #
import torch
import torchvision.models as models
import ssl
import time

ssl._create_default_https_context = ssl._create_unverified_context
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.GenerateData import MyDataset
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import random, csv
import yaml
import warnings
from Log import Logger  #

warnings.filterwarnings("ignore")
torch.manual_seed(1024)
Logger.log("./logdata/")


def read_yaml(path):
    with open(path, "r", encoding="utf8") as f:
        return yaml.safe_load(f.read())


def auc(y_true, y_pred, labels_list):

    # You need the labels to binarize
    labels = labels_list
    # Binarize ytest with shape (n_samples, n_classes)
    y_true = label_binarize(y_true, classes=labels)
    y_pred = label_binarize(y_pred, classes=labels)
    print("AUC:{:.4f}".format(roc_auc_score(y_true, y_pred, multi_class="ovo")))
    print("AUC:{:.4f}".format(roc_auc_score(y_true, y_pred, multi_class="ovr")))


def test(model, dataloader, labels_list):
    model.load_state_dict(torch.load(modelpath, map_location=torch.device("cpu")))
    model.eval()
    prob_all0 = []
    prob_all1 = []
    label_all = []
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            outputs = model(x)
            prob = outputs.cpu().numpy()
            prob_all0.extend(np.argmax(prob, axis=1))
            label_all.extend(y.cpu().numpy())
            prob_all1.extend(prob[:, 1])

    allT = classification_report(
        label_all, prob_all0, target_names=targetnames, digits=4,
    )
    print("accuracy_score:{:.4f}".format(accuracy_score(label_all, prob_all0)))
    auc(label_all, prob_all0, labels_list)
    print(allT)
    cm = confusion_matrix(label_all, prob_all0)
    print(cm)
    return 0


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device="cpu"

imgclass = "no"
try:
    Mixup = str(sys.argv[1])
    modelname = str(sys.argv[2])
    preprocess = str(sys.argv[3])
    if preprocess == "pre":
        imgclass = modelname.split("_")[0]
except:
    Mixup = "none"
    modelname = "realdataimg_baseline"
    imgclass = ";;realdataimg"
    preprocess = "pre"

print(imgclass)

all_yamldata = read_yaml("./config/train.yaml")
yamlidx = modelname.split("_")
yamldata = all_yamldata[yamlidx[0]]

num_classes = yamldata["num_classes"]
num_epochs = yamldata["num_epochs"]
data_root = yamldata["data_root"]
# data_csv = yamldata["data_csv"]
if modelname.find("adv") != -1:
    yamldata = all_yamldata["adv_data"]
data_csv = yamldata["data_csv"]
labels_list = list(range(0, num_classes))
targetnames = [str(x) for x in labels_list]

learning_rate = 0.01
batchsize = 224
num_work = 64
resize_size = 224
if modelname.find("virusdata") != -1:
    resize_size = 224
    batchsize = 224
stepsize = int(num_epochs / 4)


model_is = "densenet121"

# 模型迁移
if model_is == "resenet50":
    batchsize = 300
    model = models.resnet50(pretrained=False)
    model.conv1 = nn.Conv2d(
        1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
    )

    for param in model.parameters():
        param.requires_grad = True
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    modelpath = (
        "./pre_train_model/"
        + preprocess
        + "_"
        + modelname
        + "_"
        + Mixup
        + "_resenet50.mdl"
    )

else:
    model = models.densenet121(pretrained=False)
    model.features.conv0 = nn.Conv2d(
        1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
    )
    for param in model.parameters():
        param.requires_grad = True
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, num_classes)
    modelpath = (
        "./pre_train_model/"
        + preprocess
        + "_"
        + modelname
        + "_"
        + Mixup
        + "_densenet121.mdl"
    )


try:
    if str(sys.argv[4]) == "more":
        batchsize = resize_size * 3
        model = nn.DataParallel(model)
except:
    print("one")

print(modelpath)
print("=======learning_rate:", learning_rate, "batchsize:", batchsize, stepsize)
model = model.to(device)
#
transformed_trainset = MyDataset(data_root, resize_size, data_csv, imgclass)
trainloader = DataLoader(
    dataset=transformed_trainset,
    batch_size=batchsize,
    shuffle=True,
    num_workers=num_work,
)

transformed_valset = MyDataset(data_root, resize_size, "val", imgclass)
valloader = DataLoader(
    dataset=transformed_valset, batch_size=batchsize, shuffle=True, num_workers=num_work
)

transformed_testset = MyDataset(data_root, resize_size, "test", imgclass)
testloader = DataLoader(
    dataset=transformed_testset,
    batch_size=batchsize,
    shuffle=True,
    num_workers=num_work,
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=stepsize, gamma=0.5)
best_acc = 0


# test(model,valloader,labels_list)
# test(model, testloader, labels_list)


"""train the model"""
if 0:
    total_step = len(trainloader)
    for epoch in range(num_epochs):
        model.train()
        since = time.time()
        loss_list = []
        for i, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            is_here = "none"

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())

        if (epoch + 1) % 2 == 0:
            print(
                "Epoch [{}/{}], Step[{}/{}],Loss:{:.4f}".format(
                    epoch + 1, num_epochs, i + 1, total_step, np.mean(loss_list)
                )
            )
            print("Now  lr: ", optimizer.state_dict()["param_groups"][0]["lr"])

        lr_scheduler.step()

        correct = 0
        total = 0
        prob_all0 = []
        prob_all1 = []
        label_all = []
        if epoch % 25 == 0:
            with torch.no_grad():
                for data in trainloader:
                    images, labels = data
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    prob = outputs.cpu().numpy()
                    prob_all0.extend(np.argmax(prob, axis=1))
                    label_all.extend(labels.cpu().numpy())
                    prob_all1.extend(prob[:, 1])

            allT = classification_report(
                label_all, prob_all0, target_names=targetnames, digits=4,
            )
            print(
                "Epoch",
                epoch,
                "Accuracy of the network on the train images: %f %%"
                % (100 * correct / total),
            )
            print(allT)

        prob_all0 = []
        prob_all1 = []
        label_all = []
        correct = 0
        total = 0
        model.eval()
        with torch.no_grad():
            for data in valloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                prob = outputs.cpu().numpy()
                prob_all0.extend(np.argmax(prob, axis=1))
                label_all.extend(labels.cpu().numpy())
                prob_all1.extend(prob[:, 1])
        allT = classification_report(
            label_all, prob_all0, target_names=targetnames, digits=4,
        )
        time_elapsed = time.time() - since
        m, s = divmod(time_elapsed * (num_epochs - epoch - 1), 60)
        print(
            "Accuracy of the network on the val images: %f %%"
            % (100 * correct / total),
            "  Time:",
            int(time_elapsed),
            "s  Maybe",
            m,
            "min",
            int(s),
            "s ",
            is_here,
        )
        test_acc = correct / total
        print(allT)
        if test_acc > best_acc:
            best_epoch = epoch
            best_acc = test_acc
            print("Save ")
            torch.save(model.state_dict(), modelpath)

    print("Accuracy", best_acc, "    is", best_epoch + 1)

test(model, testloader, labels_list)
print(modelpath)

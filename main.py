import os, sys, glob, shutil, json
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import cv2

from PIL import Image
import numpy as np

from tqdm import tqdm, tqdm_notebook

#%pylab inline

import torch
torch.manual_seed(0)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset

use_cuda = torch.cuda.is_available()
print(use_cuda)

BATCHSIZE = 40
GAMMA = 0.99
lr = 1e-3

def smooth_one_hot(true_labels: np, classes: int, smoothing=0.1):
    """
    if smoothing == 0, it's one-hot method
    if 0 < smoothing < 1, it's smooth method

    """
    # true_labels = torch.full([40, 5, 11], 0)
    # true_labels[:,:,labels] = 1
    #assert 0 <= smoothing < 1
    # confidence = 1.0 - smoothing
    # label_shape = torch.Size((true_labels.size(0), classes))    # torch.Size([2, 5])
    # with torch.no_grad():
    # true_dist = np.zeros((5,11),dtype='float32')    # 空的，没有初始化
    # true_dist.fill_(smoothing / (classes - 1))
    # index = np.max(true_labels, axis=1)
    # for i in range(5):
    #     out[i][index[i]] = confidence
    # #true_dist.scatter_(1, torch.LongTensor(index.unsqueeze(1)), confidence)     # 必须要torch.LongTensor()
    # return true_dist
    assert 0 <= smoothing < 1
    confidence = 1.0 - smoothing
    label_shape = torch.Size((true_labels.size(0), classes))  # torch.Size([2, 5])
    with torch.no_grad():
        true_dist = torch.empty(size=label_shape, device=true_labels.device)  # 空的，没有初始化
        true_dist.fill_(smoothing / (classes - 1))
        _, index = torch.max(true_labels, 1)
        true_dist.scatter_(1, torch.LongTensor(index.unsqueeze(1)), confidence)  # 必须要torch.LongTensor()
    return true_dist



class SVHNDataset(Dataset):
    def __init__(self, img_path, img_label, transform=None):
        self.img_path = img_path
        self.img_label = img_label
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None

    def __getitem__(self, index):
        img = Image.open(self.img_path[index]).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        lbl = np.array(self.img_label[index], dtype=np.int)
        lbl = list(lbl) + (5 - len(lbl)) * [10]
        out = torch.full([5, 11], 0,dtype= float)
        for i in range(5):
            out[i][lbl[i]] = 1
        #out = np.array(out)
        out = smooth_one_hot(out,11,0.1)

        return img, out

    def __len__(self):
        return len(self.img_path)


train_path = glob.glob('./images/train/*.png')
train_path.sort()
train_json = json.load(open('./mchar_train.json'))
train_label = [train_json[x]['label'] for x in train_json]
#print(len(train_path), len(train_label))

train_loader = torch.utils.data.DataLoader(
    SVHNDataset(train_path, train_label,
                transforms.Compose([
                    transforms.Resize((64, 128)),
                    #transforms.RandomCrop((60, 120)),
                    transforms.ColorJitter(0.3, 0.3, 0.2),
                    #transforms.RandomRotation(10),
                    # degrees =
                    # translate = (0, 0.2)
                    # scale = (0.8, 1)
                    # fillcolor = (0, 0, 0)
                    transforms.RandomAffine(degrees=(-10, 10), translate=(0, 0.2), scale=(0.8, 1)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])),
    batch_size=40,
    shuffle=True,
    num_workers=10,
)

val_path = glob.glob('./images/val/*.png')
val_path.sort()
val_json = json.load(open('./mchar_val.json'))
val_label = [val_json[x]['label'] for x in val_json]
#print(len(val_path), len(val_label))

val_loader = torch.utils.data.DataLoader(
    SVHNDataset(val_path, val_label,
                transforms.Compose([
                    transforms.Resize((64, 128)),
                    # transforms.ColorJitter(0.3, 0.3, 0.2),
                    # transforms.RandomRotation(5),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])),
    batch_size=40,
    shuffle=False,
    num_workers=10,
)


class SVHN_Model1(nn.Module):
    def __init__(self):
        super(SVHN_Model1, self).__init__()

        model_conv = models.resnet34(pretrained=True)
        model_conv.avgpool = nn.AdaptiveAvgPool2d(1)
        model_conv = nn.Sequential(*list(model_conv.children())[:-1])
        self.cnn = model_conv
        OUTDIM = 512

        self.hd_fc1 = nn.Linear(OUTDIM, 128)
        self.hd_fc2 = nn.Linear(OUTDIM, 128)
        self.hd_fc3 = nn.Linear(OUTDIM, 128)
        self.hd_fc4 = nn.Linear(OUTDIM, 128)
        self.hd_fc5 = nn.Linear(OUTDIM, 128)
        self.dropout_1 = nn.Dropout(0.25)
        self.dropout_2 = nn.Dropout(0.25)
        self.dropout_3 = nn.Dropout(0.25)
        self.dropout_4 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(128, 11)
        self.fc2 = nn.Linear(128, 11)
        self.fc3 = nn.Linear(128, 11)
        self.fc4 = nn.Linear(128, 11)
        #self.fc5 = nn.Linear(512, 11)

    def forward(self, img):
        feat = self.cnn(img)
        # print(feat.shape)
        feat = feat.view(feat.shape[0], -1)

        feat1 = self.hd_fc1(feat)
        feat2 = self.hd_fc2(feat)
        feat3 = self.hd_fc3(feat)
        feat4 = self.hd_fc4(feat)
        feat1 = self.dropout_1(feat1)
        feat2 = self.dropout_2(feat2)
        feat3 = self.dropout_3(feat3)
        feat4 = self.dropout_4(feat4)
        c1 = self.fc1(feat1)
        c2 = self.fc2(feat2)
        c3 = self.fc3(feat3)
        c4 = self.fc4(feat4)
#        c5 = self.fc5(feat)
        return c1, c2, c3, c4
            #, c5


def train(train_loader, model, criterion, optimizer, epoch):
    # 切换模型为训练模式
    model.train()
    train_loss = []

    for i, (input, target) in enumerate(train_loader):
        if use_cuda:
            input = input.cuda()
            target = target.cuda()
        #target =  smooth_one_hot(target,11,0.1)
        c0, c1, c2, c3 = model(input)

        # , c4
        loss = criterion(c0, target[:, 0, :]) + \
               criterion(c1, target[:, 1, :]) + \
               criterion(c2, target[:, 2, :]) + \
               criterion(c3, target[:, 3, :])
               #+ \
               #criterion(c4, target[:, 4])

        # loss /= 6
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())
    return np.mean(train_loss)


def validate(val_loader, model, criterion):
    # 切换模型为预测模型
    model.eval()
    val_loss = []

    # 不记录模型梯度信息
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            if use_cuda:
                input = input.cuda()
                target = target.cuda()

            c0, c1, c2, c3 = model(input)
            #, c4
            loss = criterion(c0, target[:, 0, :]) + \
                   criterion(c1, target[:, 1, :]) + \
                   criterion(c2, target[:, 2, :]) + \
                   criterion(c3, target[:, 3, :])
                   # + \
                   # criterion(c4, target[:, 4])
            # loss /= 6
            val_loss.append(loss.item())
    return np.mean(val_loss)


def predict(test_loader, model, tta=10):
    model.eval()
    test_pred_tta = None

    # TTA 次数
    for _ in range(tta):
        test_pred = []

        with torch.no_grad():
            for i, (input, target) in enumerate(test_loader):
                if use_cuda:
                    input = input.cuda()

                c0, c1, c2, c3 = model(input)
                #, c4
                if use_cuda:
                    output = np.concatenate([
                        c0.data.cpu().numpy(),
                        c1.data.cpu().numpy(),
                        c2.data.cpu().numpy(),
                        c3.data.cpu().numpy()], axis=1)
                    # ,c4.data.cpu().numpy()
                else:
                    output = np.concatenate([
                        c0.data.numpy(),
                        c1.data.numpy(),
                        c2.data.numpy(),
                        c3.data.numpy()], axis=1)
                    # ,
                    # c4.data.numpy()
                test_pred.append(output)

        test_pred = np.vstack(test_pred)
        if test_pred_tta is None:
            test_pred_tta = test_pred
        else:
            test_pred_tta += test_pred

    return test_pred_tta

def training():

    model = SVHN_Model1()
    criterion = nn.CrossEntropyLoss()
    #optimizer = torch.optim.Adam(model.parameters(), 0.001)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                                weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=GAMMA,
                                                last_epoch=-1)
    best_loss = 1000.0

    # 是否使用GPU
    use_cuda = True
    if use_cuda:
        model = model.cuda()

    for epoch in range(20):
        train_loss = train(train_loader, model, criterion, optimizer, epoch)
        val_loss = validate(val_loader, model, criterion)

        val_label = [''.join(map(str, x)) for x in val_loader.dataset.img_label]
        val_predict_label = predict(val_loader, model, 1)
        val_predict_label = np.vstack([
            val_predict_label[:, :11].argmax(1),
            val_predict_label[:, 11:22].argmax(1),
            val_predict_label[:, 22:33].argmax(1),
            val_predict_label[:, 33:44].argmax(1),

        ]).T
        #val_predict_label[:, 44:55].argmax(1),
        val_label_pred = []
        for x in val_predict_label:
            val_label_pred.append(''.join(map(str, x[x != 10])))

        val_char_acc = np.mean(np.array(val_label_pred) == np.array(val_label))

        # val_label = [''.join(map(str, x)) for x in val_loader.dataset.img_label]
        # val_predict_label = predict(val_loader, model, 1)
        # val_predict_label = np.vstack([
        #     val_predict_label[:, :11].argmax(1),
        #     val_predict_label[:, 11:22].argmax(1),
        #     val_predict_label[:, 22:33].argmax(1),
        #     val_predict_label[:, 33:44].argmax(1),
        #
        # ]).T
        # val_predict_label[:, 44:55].argmax(1),\

        # train_label = [''.join(map(str, x)) for x in train_loader.dataset.img_label]
        # train_predict_label = predict(train_loader, model, 1)
        # train_predict_label = np.vstack([
        #     train_predict_label[:, :11].argmax(1),
        #     train_predict_label[:, 11:22].argmax(1),
        #     train_predict_label[:, 22:33].argmax(1),
        #     train_predict_label[:, 33:44].argmax(1),
        #
        # ]).T
        # train_label_pred = []
        # for x in train_predict_label:
        #     train_label_pred.append(''.join(map(str, x[x != 10])))
        #
        # train_char_acc = np.mean(np.array(train_label_pred) == np.array(train_label))

        print('Epoch: {0}, Train loss: {1} \t Val loss: {2}'.format(epoch, train_loss, val_loss))
        print('Val Acc', val_char_acc)
        # , 'Train Acc', train_char_acc
        # 记录下验证集精度
        if val_loss < best_loss:
            best_loss = val_loss
            # print('Find better model in Epoch {0}, saving model.'.format(epoch))
            torch.save(model.state_dict(), './model.pt')
        scheduler.step()



if __name__=="__main__":
    training()
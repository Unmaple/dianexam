import os
from torch.utils import data
#import transforms
import numpy as np
import torch
from torchvision.transforms import functional as F
import torchvision.transforms as tr
from PIL import Image
import cv2
import json
import show

# tivid 共5种label，每个label提取150个用作训练，30个用作检测
WIDTH = 256
Ori_bbox = 128 * 128
Target = {'0':0,'1':1, '2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9}
num_classes = 10
Trans = 1
Trainnum = 30
Testnum = 10
LOADFILE = 0


def loadjson():
    train = []
    test = []
    train_json = json.load(open('F:\steet_character_detector\data\mchar_train.json'))
    filetrain = open('trainwh.txt', 'r')
    trainwh = filetrain.readlines()
    filetest = open('testwh.txt','r')
    testwh = filetest.readlines()
    X = trainwh  # 直接每行读取
    n = len(X)
    for i in range(n):
        X[i] = X[i].strip()  # 去除后面的换行元素
        #X[i] = X[i].strip("[]")  # 去除列表的[]符号
        X[i] = X[i].split(" ")  # 根据‘，’来将字符串分割成单个元素
        X[i] = list(map(int, X[i]))
    trainwh = X
    X = testwh  # 直接每行读取
    n = len(X)
    for i in range(n):
        X[i] = X[i].strip()  # 去除后面的换行元素
        # X[i] = X[i].strip("[]")  # 去除列表的[]符号
        X[i] = X[i].split(" ")  # 根据‘，’来将字符串分割成单个元素
        X[i] = list(map(int, X[i]))
    testwh = X

    account = 0
    for x in train_json:
        #img = cv2.imread("images/train/" + x)
        width = trainwh[account][0]
        height = trainwh[account][1]
        account += 1
        # if width != img.shape[1]:
        #     print("warning1")
        # if height != img.shape[0]:
        #     print("warning1")
        train_label =list(map(int,train_json[x]['label']))
        train_height=list(map(int,train_json[x]['height']))
        train_left=list(map(int,train_json[x]['left']))
        train_width=list(map(int,train_json[x]['width']))
        train_top=list(map(int,train_json[x]['top']))
        lines = []
        for i in range(len(train_label)):
            pic_label = train_label[i]
            pic_x = (train_left[i] + train_width[i] / 2) / width
            pic_y = (train_top[i] + train_height[i] / 2) / height
            pic_width = train_width[i] / width
            pic_height = train_height[i] / height
            line = [pic_label, pic_x, pic_y, pic_width, pic_height]
            lines.append(line)
        train.append(lines)

    test_json = json.load(open('F:\steet_character_detector\data\mchar_val.json'))
    account = 0
    for x in test_json:
        #img = cv2.imread("images/val/" + x)
        width = testwh[account][0]
        height = testwh[account][1]
        account += 1
        #if width != img.shape[1]:
        #    print("warning")
        #if height != img.shape[0]:
        #    print("warning")
        test_label = list(map(int, train_json[x]['label']))
        test_height = list(map(int, train_json[x]['height']))
        test_left = list(map(int, train_json[x]['left']))
        test_width = list(map(int, train_json[x]['width']))
        test_top = list(map(int, train_json[x]['top']))
        lines = []
        for i in range(len(test_label)):
            pic_label = test_label[i]
            pic_x = (test_left[i] + test_width[i] / 2) / width
            pic_y = (test_top[i] + test_height[i] / 2) / height
            pic_width = test_width[i] / width
            pic_height = test_height[i] / height
            line = [pic_label, pic_x, pic_y, pic_width, pic_height]
            lines.append(line)
        test.append(lines)
    #loc_pic = "labels/train/" + x.split('.')[0] + '.txt'
    #pic = open(loc_pic, "w")
    #pic.close()
    return train,test


def pad_if_smaller(img, size, fill=0):
    # 将图片（torch.Tenser)填充至size大小的正方形
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, [0, 0, padw, padh], fill=fill)  # 由于报错从tuple改成了list（检修时优先注意此处）
    return img


class TvidDataset(data.Dataset):

    # TODO Implement the dataset class inherited
    # from `torch.utils.data.Dataset`.
    # tips: Use `transforms`.
    def __init__(self,root,mode):

        self.root = root
        self.mode = mode
        self.image = []
        #load = transforms.LoadImage()
        #ten = transforms.ToTensor()
        load = Image.open
        ten = F.to_tensor
        tenbbox = torch.Tensor
        # flip = transforms.RandomHorizontalFlip(1)
        # crop = transforms.RandomCrop((64,64),)  # 训练使用的数据
        # rdsize = transforms.RandomResize(0.5, 0.8)
        # Nor = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # gray = tr.Grayscale(num_output_channels=3)
        #Co = transforms.ColorJitter

        self.trainIn = torch.full([1,3,WIDTH,WIDTH],0)
        self.trainBB = []
        #self.trainLa = torch.full([Trans * Trainnum+1,num_classes],0,dtype= float)
        #self.tr = self.trainIn, self.trainLa
        # 测试使用的数据
        self.testIn = torch.full([1, 3, WIDTH, WIDTH], 0)
        self.testBB = []
        self.trainBB, self.testBB = loadjson()
        #self.testLa = torch.full([Testnum + 1,num_classes], 0,dtype= float)
        #self.te = self.testIn, self.testLa
        if LOADFILE:
            self.trainIn = torch.load("trainIn.pth")
            # self.trainBB = torch.load("trainBB.pth")
            # file = open("trainBB.txt", 'r')
            # X = file.readlines()  # 直接每行读取
            # n = len(X)
            # for i in range(n):
            #     X[i] = X[i].strip()  # 去除后面的换行元素
            #     X[i] = X[i].strip("[")  # 去除列表的[]符号
            #     X[i] = X[i].strip("]")
            #     X[i] = X[i].split(",")  # 根据‘，’来将字符串分割成单个元素
            #     X[i] = list(map(float, X[i]))
            # self.trainBB = X

            #self.trainLa = torch.load("trainLa.pth")
            self.testIn = torch.load("testIn.pth")
            # self.testBB = torch.load("testBB.pth")
            # file = open("testBB.txt", 'r')
            # X = file.readlines()  # 直接每行读取
            # n = len(X)
            # for i in range(n):
            #     X[i] = X[i].strip()  # 去除后面的换行元素
            #     X[i] = X[i].strip("[]")  # 去除列表的[]符号
            #     X[i] = X[i].split(",")  # 根据‘，’来将字符串分割成单个元素
            #     X[i] = list(map(float, X[i]))
            # self.testBB = X
            #self.testLa = torch.load("testLa.pth")


        else:
            #count = 1
            #count1 = 1
            # brightness = (0.5, 2.0)
            # contrast = (0.5, 2.0)
            # saturation = (0.5, 2.0)
            # hue = (-0.2,0.2)
            # color = tr.ColorJitter(brightness, contrast, saturation, hue)

            # degrees = (-90, 90)
            # translate = (0, 0.2)
            # scale = (0.8, 1)
            # fillcolor = (0, 0, 0)
            # TRAF = tr.RandomAffine(degrees=degrees, translate=translate, scale=scale, fillcolor=fillcolor)
            # for object in Target.keys():
            rootimg = self.root + 'images/train/'
            rootlab = self.root + 'labels/train/'

            # print(label_draft)
            # label_draft = label_draft.replace('\n',' ')
            # lines=label_draft.split()
            # print(object)
            # lines = list(map(int, lines))
            #
            # data = np.array(lines)
            # data = data.reshape(-1,5)


            for i in range(Trainnum):
                if i % 100 == 0:
                    print(i)
                file_num = str(i).zfill(6)

                #oribbox = [data[i][1], data[i][2], data[i][3], data[i][4]]
                image = load(rootimg + file_num + '.png')
                #image.show()
                # file = open(rootlab + file_num + '.txt', "r")
                # label_draft = file.read()
                # file.close()
                # boxes = []
                # lines = list(map(float,label_draft.split()))
                # for j in range(len(lines)//5):
                #     line = lines[j*5:j*5+5]
                #     #tobox = torch.tensor(line)
                #     boxes.append(line)
                #self.trainBB.append(boxes)

                #image = pad_if_smaller(image, WIDTH,0)
                image = image.resize((WIDTH, WIDTH),Image.ANTIALIAS)
                #image.show()
                #show.showing(self.trainBB[i],image)
                image1 = ten(image)
                #image1 = gray(image1)
                self.trainIn = torch.cat((self.trainIn,torch.unsqueeze(image1,dim = 0)),dim = 0)
                #print(image.size, self.trainIn[0].size,count)
                # self.trainBB.append(boxes)
                #self.trainLa[count][Target[object]-1] = 1.0
                #count += 1
                    ## unloader = tr.ToPILImage()(image1)
                    ## unloader.show()
                    #
                    # image2, bbox2 = flip(image, bbox)
                    # image2, bbox2 = ten(image2, bbox2)
                    # image2, bbox2 = Nor(image2, bbox2)
                    # self.trainIn = torch.cat((self.trainIn, torch.unsqueeze(image2, dim=0)), dim=0)
                    # # print(image.size, self.trainIn[0].size,count)
                    # self.trainBB[count] = bbox2
                    # self.trainLa[count][Target[object]-1] = 1.0
                    # count += 1

                    # for k in range(3):
                    #     image3 = color(image)
                    #     bbox3 = bbox
                    #     #image3 = transforms.pad_if_smaller(image3,128)
                    #     # image3.show()
                    #     image3, bbox3 = ten(image3, bbox3)
                    #
                    #     self.trainIn = torch.cat((self.trainIn, torch.unsqueeze(image3, dim=0)), dim=0)
                    #     # print(image.size, self.trainIn[0].size,count)
                    #     self.trainBB[count] = bbox3
                    #     self.trainLa[count][Target[object]-1] = 1.0
                    #     count += 1
                    #
                    # for k in range(3):
                    #     image4 = TRAF(image)
                    #     bbox4 = bbox
                    #     # image3 = transforms.pad_if_smaller(image3,128)
                    #     #image4.show()
                    #     image4, bbox4 = ten(image4, bbox4)
                    #     image4, bbox4 = Nor(image4, bbox4)
                    #
                    #     self.trainIn = torch.cat((self.trainIn, torch.unsqueeze(image4, dim=0)), dim=0)
                    #     # print(image.size, self.trainIn[0].size,count)
                    #     self.trainBB[count] = bbox4
                    #     self.trainLa[count][Target[object] - 1] = 1.0
                    #     count += 1



                    # image3, bbox3 = rdsize(image, bbox)
                    # image3 = transforms.pad_if_smaller(image3,128)
                    # image3, bbox3 = ten(image3, bbox3)
                    # self.trainIn = torch.cat((self.trainIn, torch.unsqueeze(image3, dim=0)), dim=0)
                    # # print(image.size, self.trainIn[0].size,count)
                    # self.trainBB[count] = bbox3
                    # self.trainLa[count][Target[object]-1] = 1.0

                    # if i % 20 == 4:
                    #     print(self.trainLa[count])
                    #
                    # count += 1


            # for i in range(151, 181):
            #     file_num = str(i).zfill(6)
            #     file_path = root1 + '\\' + file_num + '.jpeg'
            #     oribbox = [data[i][1], data[i][2], data[i][3], data[i][4]]
            #     image, bbox = load(file_path, oribbox)
            #
            #     image1, bbox1 = ten(image, bbox)
            #     #image1 = gray(image1)
            #     #image2, bbox2 = F.hflip(image)
            #     self.testIn = torch.cat((self.testIn, torch.unsqueeze(image1, dim=0)), dim=0)
            #     #print(image.size, self.trainIn[0].size, count)
            #     self.testBB[count1] = bbox1
            #     self.testLa[count1][Target[object]-1] = 1.0
            rootimg = self.root + 'images/val/'
            rootlab = self.root + 'labels/val/'
            for i in range(Testnum):
                if i % 100 == 0:
                    print(i)
                file_num = str(i).zfill(6)

                # oribbox = [data[i][1], data[i][2], data[i][3], data[i][4]]
                image = load(rootimg + file_num + '.png')
                # image.show()
                # file = open(rootlab + file_num + '.txt', "r")
                # label_draft = file.read()
                # file.close()
                # boxes = []
                # lines = list(map(float, label_draft.split()))
                # for j in range(len(lines) // 5):
                #     line = lines[j * 5:j * 5 + 5]
                #     #tobox = torch.tensor(line)
                #     boxes.append(line)
                # self.trainBB.append(boxes)

                #image = pad_if_smaller(image, WIDTH, 0)
                image = image.resize((WIDTH, WIDTH), Image.ANTIALIAS)
                # image.show()
                image1 = ten(image)
                # image1 = gray(image1)
                self.testIn = torch.cat((self.testIn, torch.unsqueeze(image1, dim=0)), dim=0)
                # print(image.size, self.trainIn[0].size,count)
                # self.testBB.append(boxes)

                # if i % 10 == 2:
                #     print(self.testLa[count1])
                #     image.show()
                #count1 += 1

                # print(image,self.trainIn[0])
                # print(file_path)

        # if self.mode == 'train':
        #     z = np.zeros((self.trainLa.size(0), num_classes), )
        #     for j in range(1, self.trainLa.size(0)):
        #         z[j][self.trainLa[j]-1] = 1
        #     # print(np.sum(X),z)
        #     self.trainLa = z
        # else :
        #     z = np.zeros((self.testLa.size(0), num_classes), )
        #     for j in range(1, self.testLa.size(0)):
        #         z[j][self.testLa[j] - 1] = 1
        #     # print(np.sum(X),z)
        #     self.testLa = z
            torch.save(self.trainIn.to(torch.device('cpu')), "trainIn.pth")
            #torch.save(self.trainBB.to(torch.device('cpu')), "trainBB.pth")

            #ipTable = ['158.59.194.213', '18.9.14.13', '58.59.14.21']
            # file = open("trainBB.txt", 'w')
            # for fp in self.trainBB:
            #     file.write(str(fp))
            #     file.write('\n')
            # file.close()

            #torch.save(self.trainLa.to(torch.device('cpu')), "trainLa.pth")
            torch.save(self.testIn.to(torch.device('cpu')), "testIn.pth")
            #torch.save(self.testBB.to(torch.device('cpu')), "testBB.pth")
            # file = open("testBB.txt", 'w')
            # for fp in self.testBB:
            #     file.write(str(fp))
            #     file.write('\n')
            # file.close()
            #torch.save(self.testLa.to(torch.device('cpu')), "testLa.pth")

        return

    def __len__(self):
        if self.mode == 'train':
            return Trainnum
        else:
            return Testnum

    def __getitem__(self,idx):
        if self.mode == 'train':
            return self.trainIn[idx+1],self.trainBB[idx]
        else:
            return self.testIn[idx+1], self.testBB[idx]




            #print(root1)
    ...

    # End of todo
if __name__ == '__main__':
    # F:\steet_character_detector\yolov5 - master\yolov5 - master\dataset\tianchi /
    dataset = TvidDataset(root=r'./', mode='train')
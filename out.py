import pandas as pd
#import glob
import os

NUM = 40000
def get(elem):
    return elem[1]
root = r'F:\steet_character_detector\yolov5-master\yolov5-master\runs\detect\exp12\labels/'
root1 = r'F:\steet_character_detector\yolov5-master\yolov5-master\dataset\tianchi\labels\val/'
#label_path=glob.glob(r'F:\steet_character_detector\yolov5-master\yolov5-master\runs\detect\exp5\labels/*.txt')
#label_path.sort()
df_submit = pd.read_csv(r'F:\steet_character_detector\data/mchar_sample_submit_A.csv')
df_submit.set_index('file_name')
correct = 0
acc = 0
for i in range(NUM):
    if i % 100 == 0:
        print(i)
    file_num = str(i).zfill(6)
    result = ''
    if os.path.exists(root + file_num + '.txt') :

        text=open(root + file_num + '.txt','r')
        result_list=[]
        for line in text.readlines():
            result_list.append((line.split(' ')[0],line.split(' ')[1]))
        result_list.sort(key=get)

        for j in result_list:
            result+=j[0]
    text.close()
    # result1 = ''
    # if os.path.exists(root1 + file_num + '.txt'):
    #
    #     text = open(root1 + file_num + '.txt', 'r')
    #     result_list = []
    #     for line in text.readlines():
    #         result_list.append((line.split(' ')[0], line.split(' ')[1]))
    #     result_list.sort(key=get)
    #
    #     for j in result_list:
    #         result1 += j[0]
    # if result == result1:
    #     correct +=1
    # text.close()
    label_path=file_num+'.png'
    df_submit['file_code'][df_submit['file_name']==label_path]=result

df_submit.to_csv('content/submit12.csv', index=None)
#acc = correct / NUM * 100
#print(correct,acc)

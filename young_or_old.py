import os
import sys

def young_or_old(classes_path,outer_path):  #年轻的返回0，年老的返回1   返回一个列表list
    with open(classes_path, 'r', encoding='UTF-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]  # 所有的年轻人15-35岁
    folderlist = os.listdir(outer_path)  # 列举文件夹名字
    list = []
    for i in folderlist:
        if i in class_names:
            list.append(0)
        else:
            list.append(1)
    return list

classes_path = os.path.expanduser(r'C:\data\yb1_young.txt')
outer_path= r"C:\data\YB1"
list= young_or_old(classes_path,outer_path) #年轻的返回0，年老的返回1   所有的年轻人15-35岁  返回一个列表list

folderlist = os.listdir(outer_path)  #列举文件夹名字

i=0
for folder in folderlist:
    inner_path = os.path.join(outer_path, folder)#文件夹路径
    filelist = os.listdir(inner_path) #列举图片名字
    for item in filelist:
        src = os.path.join(os.path.abspath(inner_path), item)  # 原图的地址
        dst = os.path.join(os.path.abspath(inner_path),  str(
            item)[:-4]+'_'+ str(list[i]) +'.bmp')  # 新图的地址（这里可以把str(folder) + '_' + str(i) + '.jpg'改成你想改的名称）
        os.rename(src,dst)
        print(dst)
    i=i+1

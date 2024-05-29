import os


outer_path= r"C:\data\YB1"
# save_txt_path=r"C:\data\txt\small_all_images.txt"
save_txt_path=r"C:\data\txt\all_images_data_expansion.txt"
f=open(save_txt_path,'w')

folderlist = os.listdir(outer_path)  #列举文件夹名字
for folder in folderlist:
    inner_path = os.path.join(outer_path, folder)#文件夹路径
    filelist = os.listdir(inner_path)  # 列举图片名字
    if (len(filelist)==10 or  len(filelist)==20):  #只要文件夹数量是10张或者20张的文件夹，因为文件夹中有些手指不全
        for item in filelist:
            label1 = item[:-4].split('_')[2]  # 去掉.bmp   性别标签
            label2 = item[:-4].split('_')[3]  # 去掉.bmp   年龄标签

            if (1 <= int(item[24:-8]) <= 10):
                path0 = inner_path + "\\" + item + ' ' + label1+' '+label2+' '+ '0'+'\n'    #0代表原图，1代表旋转90度，2代表旋转180，3代表旋转270
                path1 = inner_path + "\\" + item + ' ' + label1 + ' ' + label2 + ' ' + '1' +'\n'   # 0代表原图，1代表旋转90度，2代表旋转180，3代表旋转270
                path2 = inner_path + "\\" + item + ' ' + label1 + ' ' + label2 + ' ' + '2'+'\n'    # 0代表原图，1代表旋转90度，2代表旋转180，3代表旋转270
                path3 = inner_path + "\\" + item + ' ' + label1 + ' ' + label2 + ' ' + '3' +'\n'  # 0代表原图，1代表旋转90度，2代表旋转180，3代表旋转270
                f.write(path0)
                f.write(path1)
                f.write(path2)
                f.write(path3)
                # f.write(path)
                # print(path)




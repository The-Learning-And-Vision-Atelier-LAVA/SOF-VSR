import os
import shutil

pth = 'TIP/data/train'

directory = os.listdir(pth)

print(directory)

# for i in directory: #lr 바꾸기
#     print(i)
#     subdir = os.listdir(pth+'/'+i)
#
#     lrdir = os.listdir(pth+'/'+i+'/'+subdir[1])
#
#     for j in range(len(lrdir)):
#         os.rename(pth+'/'+i+'/'+subdir[1]+'/'+lrdir[j], pth+'/'+i+'/'+subdir[1]+'/'+"lr"+str(j)+'.png')
#         print(pth+'/'+i+'/'+subdir[1]+'/'+"lr"+str(j)+'.png')


# for i in directory: #hr 바꾸기
#     print(i)
#     subdir = os.listdir(pth+'/'+i)
#
#     lrdir = os.listdir(pth+'/'+i+'/'+subdir[0])
#
#     for j in range(len(lrdir)):
#         os.rename(pth+'/'+i+'/'+subdir[0]+'/'+lrdir[j], pth+'/'+i+'/'+subdir[0]+'/'+"hr"+str(j)+'.png')
#         print(pth+'/'+i+'/'+subdir[0]+'/'+"hr"+str(j)+'.png')


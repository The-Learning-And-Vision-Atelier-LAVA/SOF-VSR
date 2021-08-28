import os
import shutil

pth = 'TIP/data/train'

directory = os.listdir(pth)

print(directory)


for i in directory: #hr 바꾸기

    if i == 'GirlRunningOnGrass' or i == 'ChefCooking1' or i =='TreeTrunkMoving':
        subdir = os.listdir(pth+'/'+i)
        print(subdir)
        lrdir = os.listdir(pth+'/'+i+'/'+subdir[0])
        print(pth+'/'+i+'/'+subdir[0])
        print(lrdir)

        for j in range(len(lrdir)):
            fname = lrdir[j][:-4]
            fnum = fname[2:]
            print(fname)
            os.rename(pth+'/'+i+'/'+subdir[0]+'/'+fname+'.png', pth+'/'+i+'/'+subdir[0]+'/'+'hr'+fname+'.png')
            print(pth+'/'+i+'/'+subdir[0]+'/'+fname+'.png'+' -> ' + pth+'/'+i+'/'+subdir[0]+'/'+'hr'+fname+'.png')

    else:
        pass

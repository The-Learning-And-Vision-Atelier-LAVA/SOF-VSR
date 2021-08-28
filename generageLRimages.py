import cv2
import os

pth = 'TIP/data/train'

directory = os.listdir(pth)

print(directory)

for i in directory:
    if i == 'GirlRunningOnGrass' or i == 'ChefCooking1' or i == 'TreeTrunkMoving':
        subdir = os.listdir(pth + '/' + i)

        lrdir = os.listdir(pth + '/' + i + '/' + subdir[0])

        for j in range(len(lrdir)):
            img = pth + '/' + i + '/' + subdir[0] + '/' + lrdir[j]
            src = cv2.imread(img, cv2.IMREAD_COLOR)
            dst = cv2.resize(src, dsize=(0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)

            fname = lrdir[j][:-4]
            fnum = fname[2:]
            name = pth + '/' + i + '/' + subdir[1] + '/lr' + fnum + '.png'
            print(name)
            cv2.imwrite(name, dst)

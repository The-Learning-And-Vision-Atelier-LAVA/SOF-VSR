import os
import shutil

pth = 'D:/TVD/TVD_960/hevc/frame'

directory = os.listdir(pth)

print(directory)

for i in directory:  # hr 바꾸기

    subdir = os.listdir(pth + '/' + i)
    print(subdir)
    lrdir = os.listdir(pth + '/' + i + '/' + subdir[1])
    print(pth + '/' + i + '/' + subdir[1])
    print(lrdir)

    for j in range(len(lrdir)):  # hr directory
        origname = lrdir[j][:-4]

        fname = origname[3:]
        print(fname)

        if fname == origname[3:]:
            os.rename(pth + '/' + i + '/' + subdir[1] + '/' + origname + '.png',
                      pth + '/' + i + '/' + subdir[1] + '/' + 'hr' + str(fname) + '.png')
            print(pth + '/' + i + '/' + subdir[1] + '/' + origname + '.png' + ' -> ' + pth + '/' + i + '/' + subdir[
                1] + '/' + 'hr' + str(fname) + '.png')

import os
import shutil

pth = 'TIP/data/train/TVD_HEVC_frame'

directory = os.listdir(pth)

print(directory)


for i in directory: #hr 바꾸기

    subdir = os.listdir(pth+'/'+i)
    print(subdir)
    lrdir = os.listdir(pth+'/'+i+'/'+subdir[0])
    print(pth+'/'+i+'/'+subdir[0])
    print(lrdir)

    for j in range(len(lrdir)): # hr directory
        origname = lrdir[j][:-4]

        fname = origname
        fname = fname[-5:]

        if fname[0] == '_':
            fname = fname[4]
            print(fname)

        else:
            fname = fname[3:]
            print(fname)

        os.rename(pth+'/'+i+'/'+subdir[0]+'/'+origname+'.png', pth+'/'+i+'/'+subdir[0]+'/'+'hr'+str(int(fname)-1)+'.png')
        print(pth+'/'+i+'/'+subdir[0]+'/'+origname+'.png'+' -> ' + pth+'/'+i+'/'+subdir[0]+'/'+'hr'+str(int(fname)-1)+'.png')

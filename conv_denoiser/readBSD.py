# This code belongs to the paper
# 
# J. Hertrich, S. Neumayer and G. Steidl.  
# Convolutional Proximal Neural Networks and Plug-and-Play Algorithms.
# arXiv Preprint#2011.02281, 2020.
#
# Please cite the paper if you use this code.
#
# This file contains a method to crop patches from the BSDS data set 
#
from PIL import Image
import os
import numpy as np


def loadFromPath(noise_level=25,patch_size=40,shift=None,path='train_BSDS_png',rotate=False):
    if shift is None:
        shift=patch_size//2
    patches=[]
    fileList=os.listdir(path)
    fileList.sort()
    for fileName in fileList:
        img=Image.open(path+'/'+fileName)
        img=img.convert('L')
        img_gray=1.0*np.array(img)
        img_gray/=255.0
        img_gray-=.5
        up=0
        left=0
        while up+patch_size<=img_gray.shape[0]:
            while left+patch_size<=img_gray.shape[1]:
                patch=img_gray[up:(up+patch_size),left:(left+patch_size)]
                patches.append(patch)
                if rotate:
                    patch=np.rot90(patch)
                    patches.append(patch)
                    patch=np.rot90(patch)
                    patches.append(patch)
                    patch=np.rot90(patch)
                    patches.append(patch)
                left+=shift
            up+=shift
            left=0
    patches=np.array(patches)
    patches_noisy=patches+noise_level/255.0*np.random.normal(size=patches.shape)
    return patches,patches_noisy

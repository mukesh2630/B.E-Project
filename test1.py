import glob
import os
import numpy as np
import  cv2
#from try1 import *

test_path = 'C:/Users/mukes/Desktop/TensorFlow Speech Recognition Challenge/picts/test/'

def load_images_from_folder(folder):
    imgs = []
    fnames = []
    for filepath in glob.glob(test_path + '/*.png'):
        #print(filepath)
        img = cv2.imread(filepath)
        img = cv2.resize(img, (128, 128))
        imgs.append(img)
        #print(imgs)
        fnames.append(filepath.split('\\')[-1].split('.png')[0])
        #print(fnames)
    return imgs, fnames

imgs, fnames = load_images_from_folder(test_path)

print(len(imgs))
print(len(fnames))





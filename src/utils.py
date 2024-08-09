import json
import numpy as np
import torchvision
import cv2
import os
import torchvision.transforms.functional as F
import albumentations as A
from torchvision.utils import make_grid
from PIL import Image


#Reads in a trainConfig.json from the root directory.
#If it isn't found asks the user for a path to the config json
def ReadConfig(configPath):
    file = None
    try:
        file = open(configPath)
    except Exception:
        flag = True
        while(flag):
            path = input("No trainConfig.json found. Enter the path to a config file:")

            try:
                file = open(path)
                flag = False
            except:
                line = input('Path not valid. Try again? y/n:')
                if(line.lower()[0] == 'y'):
                    flag = True
                else:
                    flag = False
    configDict = json.load(file)
    return configDict


#Scales, normalizes and converts an image to a tensor before sending it to the GPU
#images: list of images to transform
#dimentions the dimentions to scale to. Should be the same as the model was trained on
#device: the device to send the image to. Should be a GPU
def TransformImages(images,dimentions,device):
    transform = A.Compose(
    [
        A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        A.Resize(height=dimentions, width=dimentions, p=1),   
    ],
    p=1.0,)
    ret = list()
    for image in images:

        transformed = transform(image=image)
        
        image = transformed['image']

        torchImage = torchvision.tv_tensors.Image(image)
        torchImage = np.transpose(torchImage, axes=(2, 0, 1))
        torchImage = torchImage.to(device)
        ret.append(torchImage)
    return ret

#Saves images in pairs to bmp files
#imageList: list of imaes to save each pair of images (0 and 1) (2 and 3) are merged into one image
#imageNames: the names to give the images
#location: the folder to save images in
def SaveImages(imageList,imageNames,location):
    i = 0
    while i < len(imageList):
        sublist = [imageList[i],imageList[i+1]]
        F.to_pil_image(make_grid(sublist,nrow=2)).save(location+imageNames[int(i/2)]+'.bmp')
        i = i+2

#loads images form a folder and returns a list of images and a list of image names
def LoadImagesFromFolder(folder):
    images = []
    imageNames = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename),flags=cv2.IMREAD_COLOR+cv2.IMREAD_IGNORE_ORIENTATION)
        if img is not None:
            images.append(img)
            imageNames.append(filename)
    return images,imageNames



            
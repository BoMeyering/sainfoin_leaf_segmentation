import os
import pandas as pd
import json
from torchvision.io import read_image
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from PIL import Image, ImageChops
from torchvision.utils import draw_bounding_boxes
import torchvision.tv_tensors._mask


plt.rcParams["savefig.bbox"] = 'tight'
def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        img.show
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])



def show_img(img):
     cv2.namedWindow('test', cv2.WINDOW_NORMAL)
     cv2.imshow('test', img)
     cv2.waitKey()
     cv2.destroyAllWindows()


def splitData(rgbJson):
    rgbDict = json.load(open(rgbJson,'r'))
    mapDict = dict()
    length = len(rgbDict)
    i=0
    for id in rgbDict:
        mapDict[i] = id
        i = i+1

    fullArrray = np.arange(0,length)
    splits = train_test_split(fullArrray, test_size=.2, train_size=.8, random_state=None, shuffle=True, stratify=None)
    return mapDict, splits[0], splits[1]
    
mapDict,trainArray,validationArray = splitData('data/processed/rgbPairs.json')
#print(validationArray)

class customDataset(torch.utils.data.Dataset):
    def __init__(self,boundingBoxes,rgbJson,imageFolder, indexSubset, mapDict):
        #read CSV to a panda with column names
        self.boundingBoxes = pd.read_csv(boundingBoxes,names=['id','index','x1','y1','x2','y2'])
        self.rgbPairs = json.load(open(rgbJson,'r'))
        self.imageFolder = imageFolder
        self.subSet = indexSubset
        self.mapDict = mapDict
    def __len__(self):
        return len(self.subSet)
    def __getItem__(self,idx):
        #given an index look up the corosponding image
        #make a tuple of the image and a dictionary containing:
        #boxes from CSV
        #

        #convert the validation index to an index from the full array so the id can be retrieved
        trueIndex = self.subSet[idx]
        #get the id of image maching the index
        id = self.mapDict[trueIndex]
        #read in the image for openCV and torch vision
        image = cv2.imread(self.imageFolder + id + '.png')
        torchImage = read_image(self.imageFolder + id + '.png')
        #search for the bounding boxes associated with the id
        matches = (self.boundingBoxes.loc[self.boundingBoxes['id'].str.contains(id)])
        #preallocate the tensor using zeros
        boxTensor = torch.zeros(len(matches),4)
        #use a loop to fill the tensor using the colums of the matches dataframe
        #note that the row indicies of the dataframe are the same as in the origional CSV and are
        #threfore require n to be used as an iterator variable for the box tensor
        n = 0
        for i,row in matches.iterrows():
            boxTensor[n][0] = row['x1']
            boxTensor[n][1] = row['y1']
            boxTensor[n][2] = row['x2']
            boxTensor[n][3] = row['y2']
            n = n+1

        #1D tensor of lables corosponding to the boxes and masks initilized to zeros
        labelTensor = torch.zeros(len(boxTensor))

        #pre access the dictionary corosponding to the image id
        indexDic = self.rgbPairs[id]
        print(id)
        print(indexDic)

        #stores the mask images untill they are converted to a tensor
        maskStack = list()

        #Use n to access tensors during each iteratiion
        n = 0
        for key in indexDic:
            #get the value associated with the key
            strLabel = indexDic[key]['class']
            #assign an integer lable to each class
            intLabel = 0
            if strLabel == 'leaflet': intLabel = 1
            elif strLabel == 'petiole': intLabel = 2
            elif strLabel == 'folded_leaflet': intLabel = 3
            elif strLabel == 'pinched_leaflet': intLabel = 4
            #store the class in the label tensor at index n
            labelTensor[n] = intLabel
            

            #get the color of the anotation
            rgb = indexDic[key]['rgb']
            bgr = rgb[::-1]
            bgr = np.array(bgr,dtype=np.uint8)
            # create a black and white mask image that contains only the anotation and add it to the list
            mask = cv2.inRange(image, bgr, bgr)
            maskStack.append(mask)

            n = n+1
        print(labelTensor)


        #make the unique image id the same as the index
        image_id = idx
        #calculate the area of the box tensor for the area tensor
        area = (boxTensor[:, 3] - boxTensor[:, 1]) * (boxTensor[:, 2] - boxTensor[:, 0])
        #assume all instances are not crowd
        iscrowd = torch.zeros((len(boxTensor),), dtype=torch.int64)

        #convert the list of images to an array then to a tensor
        maskArray = np.array(maskStack)
        maskTensor = torchvision.tv_tensors.Mask(maskArray)

        #assemble the target dictionary
        target = {}
        target['boxes'] = boxTensor
        target['masks'] = maskTensor
        target['labels'] = intLabel
        target['image_id'] = image_id
        target['area'] = area
        target['iscrowd'] = iscrowd

        return torchImage, target

vds = customDataset('data/processed/boundingBoxesBackup.csv','data/processed/rgbPairs.json','data/raw/segmentedImages/',validationArray,mapDict)
#print(vds.__len__())

tds = customDataset('data/processed/boundingBoxesBackup.csv','data/processed/rgbPairs.json','data/raw/segmentedImages/',trainArray,mapDict)

def showItem(index, dataset):
    item = dataset.__getItem__(index)

    #show the color mask with bounding boxes
    result = draw_bounding_boxes(item[0][:3], item[1]['boxes'], width=5)
    F.to_pil_image(result).show()
    img = F.to_pil_image(item[1]['masks'][0])

    #show the black and white tensor mask 
    for i in item[1]['masks']:
        img = ImageChops.add(img,F.to_pil_image(i))
        #input("Press enter to continue")
    img.show()


for i in range(len(tds)):
    showItem(i,tds)
    input("Press enter for next photo")

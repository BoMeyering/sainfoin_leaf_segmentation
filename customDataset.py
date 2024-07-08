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
from PIL import Image
from torchvision.utils import draw_bounding_boxes


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

class ValidatinDataset(torch.utils.data.Dataset):
    def __init__(self,boundingBoxes,rgbJson,imageFolder, validationIndexes, mapDict):
        #read CSV to a panda with column names
        self.boundingBoxes = pd.read_csv(boundingBoxes,names=['id','index','x1','y1','x2','y2'])
        self.rgbPairs = json.load(open(rgbJson,'r'))
        self.imageFolder = imageFolder
        self.subSet = validationIndexes
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

        labelTensor = torch.zeros(len(boxTensor))

        
        maskTensor = torch.zeros(len(boxTensor),image.shape[0],image.shape[1])
        indexDic = self.rgbPairs[id]
        print(id)
        print(indexDic)

        n = 0
        for key in indexDic:
            strLabel = indexDic[key]['class']
            intLabel = 0
            if strLabel == 'leaflet': intLabel = 1
            elif strLabel == 'petiole': intLabel = 2
            elif strLabel == 'folded_leaflet': intLabel = 3
            elif strLabel == 'pinched_leaflet': intLabel = 4
            labelTensor[n] = intLabel
            

            #get the color of the anotation
            rgb = indexDic[key]['rgb']
            bgr = rgb[::-1]
            bgr = np.array(bgr,dtype=np.uint8)
            # create a black and white mask image that contains only the anotation
            mask = cv2.inRange(image, bgr, bgr)
            imageTensor = torch.Tensor(mask)
            maskTensor[n] = imageTensor

            n = n+1
        print(labelTensor)

        # for i in range(1,len(boxTensor)):
        #     strLabel = indexDic[str(i)]
        #     intLabel = 0
        #     if strLabel == 'leaflet': intLabel = 1
        #     elif strLabel == 'petiole': intLabel = 2
        #     elif strLabel == 'folded_leaflet': intLabel = 3
        #     elif strLabel == 'pinched_leaflet': intLabel = 4
        #     labelTensor[i-1] = intLabel
        

        # maskTensor = torch.Tensor()

        # image = cv2.imread(self.imageFolder + id + '.png')
        # #for each index AKA annotation
        # for index in self.rgbPairs[id]:
        #     #get the color of the anotation
        #     rgb = self.rgbPairs[id][index]['rgb']
        #     bgr = rgb[::-1]
        #     bgr = np.array(bgr,dtype=np.uint8)
        #     # create a black and white mask image that contains only the anotation
        #     mask = cv2.inRange(image, bgr, bgr)
        #     imageTensor = torch.Tensor(mask)
        #     maskTensor.cat(imageTensor)

        image_id = idx
        area = (boxTensor[:, 3] - boxTensor[:, 1]) * (boxTensor[:, 2] - boxTensor[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((len(boxTensor),), dtype=torch.int64)

        
        target = {}
        target['boxes'] = boxTensor
        target['masks'] = maskTensor
        target['labels'] = intLabel
        target['image_id'] = image_id
        target['area'] = area
        target['iscrowd'] = iscrowd

        return torchImage, target
            

        return 0
vds = ValidatinDataset('data/processed/boundingBoxes.csv','data/processed/rgbPairs.json','data/raw/segmentedImages/',validationArray,mapDict)
#print(vds.__len__())
item = vds.__getItem__(4)

result = draw_bounding_boxes(item[0][:3], item[1]['boxes'], width=5)
F.to_pil_image(result).show()
show(result)


class TrainingDataset(torch.utils.data.Dataset):
    def __init__(self, annotations_file, img_dir):
        #in format ID,index,x1,x2,x3,x4
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
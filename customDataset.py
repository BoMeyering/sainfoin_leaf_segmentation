import os
import pandas as pd
import json
from torchvision.io import read_image
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
import torch
import torchvision


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
print(validationArray)

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
        #read in the image
        image = cv2.imread(self.imageFolder + id + '.png')
        #search for the bounding boxes associated with the id
        matches = (self.boundingBoxes.loc[self.boundingBoxes['id'].str.contains(id)])
        #preallocate the tensor using zeros
        boxTensor = torch.zeros(len(matches),4)
        #use a loop to fill the tensor using the colums of the matches dataframe
        #note that the row indicies of the dataframe are the same as in the origional CSV and are
        #threfore require n to be used as an iterator variable fo the box tensor
        n = 0
        print(matches['x1'])
        for i,row in matches.iterrows():
            print(row['x1'])
            boxTensor[n][0] = row['x1']
            boxTensor[n][1] = row['y1']
            boxTensor[n][2] = row['x2']
            boxTensor[n][3] = row['y2']
            n = n+1
        return boxTensor
            

        return 0
vds = ValidatinDataset('data/processed/boundingBoxes.csv','data/processed/rgbPairs.json','data/raw/segmentedImages/',validationArray,mapDict)
print(vds.__len__())
print(vds.__getItem__(14))

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
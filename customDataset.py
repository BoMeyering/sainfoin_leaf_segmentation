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
import albumentations as A


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

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self,boundingBoxes,rgbJson,imageFolder, maskFolder, indexSubset, mapDict):
        #read CSV to a panda with column names
        self.boundingBoxes = pd.read_csv(boundingBoxes,names=['id','index','x1','y1','x2','y2'])
        self.rgbPairs = json.load(open(rgbJson,'r'))
        self.imageFolder = imageFolder
        self.maskFolder = maskFolder
        self.subSet = indexSubset
        self.mapDict = mapDict
    def __len__(self):
        return len(self.subSet)
    def __getitem__(self,idx):
        #given an index look up the corosponding image
        #make a tuple of the image and a dictionary containing:
        #boxes from CSV
        #

        #convert the validation index to an index from the full array so the id can be retrieved
        trueIndex = self.subSet[idx]
        #get the id of image maching the index
        id = self.mapDict[trueIndex]
        #read in the maskimage image for openCV
        maskImage = cv2.imread(self.maskFolder + id + '.png')

        image = cv2.imread(self.imageFolder + self.rgbPairs[id]['externalID'])
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
        #print(indexDic)

        #stores the mask images untill they are converted to a tensor
        maskList = list()

        #Use n to access tensors during each iteratiion
        n = 0
        for key in indexDic:
            try:
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
                mask = cv2.inRange(maskImage, bgr, bgr)
                maskList.append(mask)

                n = n+1
            except:
                i = 0


        target_img_size = 1024

        transform = A.Compose(
        [
            A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            A.Resize(height=target_img_size, width=target_img_size, p=1),
            A.HorizontalFlip(p=0.5),
            #A.ColorJitter(),
            #A.ChannelShuffle(),
            A.GaussianBlur(p=.2),
            A.Affine(translate_percent=(-.10, .10), rotate=(-30, 30), shear=(-15, 15)),
            
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format="pascal_voc", min_area=0, min_visibility=0, label_fields=["labels"]
        ),)
        
        transformed = transform(
            image=image,
            masks=maskList,
            bboxes=boxTensor,
            labels = labelTensor)
        
        image = transformed['image']
        #show_img(image)
        maskList = transformed['masks']
        boxTensor = transformed['bboxes']
        labelTensor = transformed['labels']

        area = torch.zeros(len(boxTensor))
        for i in range(len(boxTensor)):
            area[i] = boxTensor[i][2]-boxTensor[i][0] * (boxTensor[i][1] -boxTensor[i][3])

        #make the unique image id the same as the index
        image_id = idx
        #calculate the area of the box tensor for the area tensor
        #area = (boxTensor[:, 3][0] - boxTensor[:, 1][0]) * (boxTensor[:, 2][0] - boxTensor[:, 0][0])
        #assume all instances are not crowd
        iscrowd = torch.zeros((len(boxTensor),), dtype=torch.int64)

        boxArray = np.array(boxTensor)
        boxTensor = torch.tensor(boxArray)
        labelArray = np.array(labelTensor)
        labelTensor = torch.tensor(labelArray,dtype=torch.int64)

        #convert the list of images to an array then to a tensor
        maskArray = np.array(maskList)
        maskTensor = torchvision.tv_tensors.Mask(maskArray)

        torchImage = torchvision.tv_tensors.Image(image)
        torchImage = np.transpose(torchImage, axes=(2, 0, 1))

        #assemble the target dictionary
        target = {0: 'test'}
        target["boxes"] = boxTensor
        target['masks'] = maskTensor
        target['labels'] = labelTensor
        target['image_id'] = image_id
        target['area'] = area
        target['iscrowd'] = iscrowd

        return torchImage, target

# vds = CustomDataset('data/processed/boundingBoxesBackup.csv','data/processed/rgbPairs.json','data/raw/origionalImages/','data/raw/segmentedImages/',validationArray,mapDict)

# tds = CustomDataset('data/processed/boundingBoxesBackup.csv','data/processed/rgbPairs.json','data/raw/origionalImages/','data/raw/segmentedImages/',trainArray,mapDict)
# img, item = tds[1]
# print('image:' + str(img.shape))
# print('boxes:' + str(item['boxes'].shape))
# print('masks:' + str(item['masks'].shape))
# print('labels:' + str(item['labels'].shape))
# print('image_id:' + str(item['image_id']))
# print('area:' + str(item['area'].shape))
# print('iscrowd' + str(item['iscrowd'].shape))

# def showItem(index, dataset):
#     item = dataset[index]

#     #show the color mask with bounding boxes
#     #result = draw_bounding_boxes(item[0][:3], item[1]['boxes'], width=5)
#     #F.to_pil_image(result).show()

#     #F.to_pil_image(item[0]).show()

#     img = F.to_pil_image(item[1]['masks'][0])
#     #show the black and white tensor mask 
#     for i in item[1]['masks']:
#         img = ImageChops.add(img,F.to_pil_image(i))
#         #input("Press enter to continue")
#     img.show()


# for i in range(len(tds)):
#     showItem(i,tds)
#     input("Press enter for next photo")

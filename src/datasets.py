import pandas as pd
import json
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
import torch
import torchvision.tv_tensors._mask
import albumentations as A





#splits the images into vaidation and training sets
def splitData(rgbJson):
    #open the rgbJson file
    rgbDict = json.load(open(rgbJson,'r'))
    #mapDict maps indexes to ids that can be used to look up an image in the rgbJson file
    mapDict = dict()
    length = len(rgbDict)
    i=0
    for id in rgbDict:
        mapDict[i] = id
        i = i+1
    #make an array with numbers form 0 to the the number of images
    fullArrray = np.arange(0,length)
    #split the data so 15% is for validation and 85% is used for training 
    splits = train_test_split(fullArrray, test_size=.15, train_size=.85, random_state=None, shuffle=True, stratify=None)
    return mapDict, splits[0], splits[1]
    

#CustomDataset is used for both the training and validation data sets
#It alows an index to retrieve tensor versions of:
#   Bounding boxes
#   Image masks
#   Image Labels
#This makes it a valid input to a data loader wich is then passed to the model
#Custom Dataset also handles making augmentations to the images, masks, and bounding boxes
class CustomDataset(torch.utils.data.Dataset):
    #boundingBoxes: filepath to the boundingBox csv
    #rgbJson: path to the rgbJson
    #imageFolder: path to the origional images
    #maskFolder: path the the mask folder
    #indexSubset: an arrray containing which indices from the whole dataset should be included in this one
    #   used for training validation spliting
    #mapDict: dictionary maping indicies to image IDs
    #validation: if true augmentations are disabled
    def __init__(self,boundingBoxes,rgbJson,imageFolder, maskFolder, indexSubset, mapDict, validation=False, imageSize = 1024):
        #read CSV to a panda with column names
        self.boundingBoxes = pd.read_csv(boundingBoxes,names=['id','index','x1','y1','x2','y2'])
        self.rgbPairs = json.load(open(rgbJson,'r'))
        self.imageFolder = imageFolder
        self.maskFolder = maskFolder
        self.subSet = indexSubset
        self.mapDict = mapDict
        self.validation = validation
        self.targetImageSize = imageSize
    def __len__(self):
        return len(self.subSet)
        #return 24
    def __getitem__(self,idx):

        #convert the validation index to an index from the full array so the id can be retrieved
        trueIndex = self.subSet[idx]
        #get the id of image maching the index
        id = self.mapDict[trueIndex]
        #read in the maskimage image for openCV
        maskImage = cv2.imread(self.maskFolder + id + '.png')
        #get the origional image name form rgbJson and read the image ignoreing metadata orientation
        image = cv2.imread(self.imageFolder + self.rgbPairs[id]['externalID'], flags= cv2.IMREAD_COLOR + cv2.IMREAD_IGNORE_ORIENTATION)
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
        indexDict = self.rgbPairs[id]

        #stores the mask images untill they are converted to a tensor
        maskList = list()

        #Use n to access tensors during each iteratiion
        n = 0
        for key in indexDict:
            #will throw an erro when reading the externalID
            #ignore the error and move on
            try:
                #get the value associated with the key
                strLabel = indexDict[key]['class']
                #assign an integer label to each class
                intLabel = 0
                if strLabel == 'leaflet': intLabel = 1
                elif strLabel == 'petiole': intLabel = 0
                elif strLabel == 'folded_leaflet': intLabel = 2
                elif strLabel == 'pinched_leaflet': intLabel = 2

                #store the class in the label tensor at index n
                labelTensor[n] = intLabel
                
                #get the color of the anotation
                rgb = indexDict[key]['rgb']
                bgr = rgb[::-1]
                bgr = np.array(bgr,dtype=np.uint8)
                # create a black and white mask image that contains only the anotation and add it to the list
                mask = cv2.inRange(maskImage, bgr, bgr)
                maskList.append(mask)

                n = n+1
            except:
                i = 0

        #set the size to scale images to 1024 works well, trains faster with 512
        #if the dataset is for validation only normalize and rescale the image
        if self.validation:
            transform = A.Compose(
            [
                A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                A.Resize(height=self.targetImageSize, width=self.targetImageSize, p=1),   
            ],
            p=1.0,
            # is_check_shapes=False,
            bbox_params=A.BboxParams(
                format="pascal_voc", min_area=0, min_visibility=0, label_fields=["labels"]
            ),)
        # if the image is for training apply various augmentations
        else:
            transform = A.Compose(
            [
                A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                A.Resize(height=self.targetImageSize, width=self.targetImageSize, p=1),
                A.HorizontalFlip(p=0.5),
                #probably redundent with A.Affine
                A.RandomRotate90(p=.5),
                A.ColorJitter(),
                #A.ChannelShuffle(),
                A.GaussianBlur(p=.2),
                A.Affine(translate_percent=(-.10, .10), rotate=(-90, 90), shear=(-15, 15)),
                
            ],
            p=1.0,
            # is_check_shapes=False,
            bbox_params=A.BboxParams(
                format="pascal_voc", min_area=0, min_visibility=0, label_fields=["labels"]
            ),)
        #run the augmentations
        transformed = transform(
            image=image,
            masks=maskList,
            bboxes=boxTensor,
            labels = labelTensor)
        
        #split the augmentiton outputs into seperate variables
        image = transformed['image']
        #show_img(image)
        maskList = transformed['masks']
        boxTensor = transformed['bboxes']
        labelTensor = transformed['labels']

        #calculat the area of each of the bounding boxes
        area = torch.zeros(len(boxTensor))
        for i in range(len(boxTensor)):
            area[i] = boxTensor[i][2]-boxTensor[i][0] * (boxTensor[i][1] -boxTensor[i][3])

        #make the unique image id the same as the index
        image_id = idx

        #assume all instances are not crowd
        iscrowd = torch.zeros((len(boxTensor),), dtype=torch.int64)

        #fix the formating of the tensors by converting them to np arrays and back
        boxArray = np.array(boxTensor)
        boxTensor = torch.tensor(boxArray)
        labelArray = np.array(labelTensor)
        labelTensor = torch.tensor(labelArray,dtype=torch.int64)

        #convert the list of images to an array then to a tensor
        maskArray = np.array(maskList)
        #make the mask 0 and 1 instead of 0 and 255
        maskArray = maskArray/255
        #maskTensor = torchvision.tv_tensors.Mask(maskArray)
        maskTensor = torch.tensor(maskArray,dtype=torch.uint8)

        #convert the image to a tensor and re arrange the channels
        torchImage = torchvision.tv_tensors.Image(image)
        torchImage = np.transpose(torchImage, axes=(2, 0, 1))

        #assemble the target dictionary
        target = {}
        target["boxes"] = boxTensor
        target['masks'] = maskTensor
        target['labels'] = labelTensor
        target['image_id'] = image_id
        target['area'] = area
        target['iscrowd'] = iscrowd
        target['imageID'] = id

        
        return torchImage, target

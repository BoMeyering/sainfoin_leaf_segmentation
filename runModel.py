import customDataset as cd
import torchvision.tv_tensors
import customDataset as cd
import torch
import torchvision
from torchvision.utils import draw_bounding_boxes, make_grid
import torchvision.transforms.functional as F
from ensemble_boxes import *
import random as R
import cv2
import torchvision.tv_tensors
import numpy as np
from torchvision.utils import draw_bounding_boxes
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import os
import albumentations as A
import json
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.rpn import RPNHead



#Loads a trianed model from a file
def get_model_instance_segmentation(checkpoint):

    model = torch.load(checkpoint)
    model.eval()
    
    return model


#filters the output of the model using ZFTurbo's weight-boxes-fusion
#minScore: minimum score for a bounding box to be concidered
#minIou: how much bounding boxes can overlap
#maskThreash: removes bad masks .3 to .7 works well
#imgDimetnions: the width and height of the images
#rainbow: if true the each segment of the output will be assigned a random color
def filterOutput(predictions,images,minScore,minIou,maskThreash,maskOverlap,imgDimentions,rainbow=False):
    #the list of images to be saved, each pair of images is combined into a single image before saveing
    imageList = list()
    #the color to use of different classes
    colors = [[255,0,0],[0,255,0],[0,0,255],[255,232,0],[0,128,128]]
    #for each prediciton fitler an save images
    for i in range(len(predictions)):
        pred = predictions[i]

 
        #remove low confidence masks
        masks = (pred["masks"] > maskThreash).squeeze(1)
        #load the scores bounding boxes and labels
        scores = pred['scores']
        boxes = pred['boxes']
        labels = pred['labels']

        #rescale to bounding boxes to work with weighted boxes fusion filtering
        boxes = boxes/imgDimentions

        #filter the bounding boxes
        boxes, newScores, labels = weighted_boxes_fusion([boxes], [scores], [labels], weights=None, iou_thr=minIou, skip_box_thr=minScore)

        #make an new mask list with only the masks left after bounding box filtering in order of confidence
        newScoreIndex = 0
        maskList = []
        for j in range(len(scores)):
            if scores[j] == newScores[newScoreIndex]:
                newScoreIndex = newScoreIndex + 1
                maskList.append(masks[j])
            if newScoreIndex == len(newScores):
                break
        
        #draw the bounding boxes on the origional image and add it to the image list
        img = torch.tensor(images[i],dtype=torch.uint8)
        imageList.append(draw_bounding_boxes(img,torch.tensor(boxes*imgDimentions,dtype=torch.int64),width=5))
        
        #The color mask that will be displayed at the end
        compositMask = torch.zeros(3,imgDimentions,imgDimentions)
        #tracks what parts of the mask already have data in them AKA not background
        #this allows for masks to be added in order from most to least conficence
        #excludeing a mask if it overlaps with a higher conficence mask too much
        binMask = np.zeros([imgDimentions,imgDimentions],dtype=np.int8)

        #loop over all the predicted masks
        for j in range(min(len(labels),len(masks))):

            #convert to numpy arrays and compare to the existing mask to check for overlap
            masks = masks.to('cpu')
            npMask = np.array(masks[j])
            binMaskOnes = np.count_nonzero(binMask == 1)
            evalMask = np.add(npMask,binMask,dtype=np.int8)
            evalOnes = np.count_nonzero(evalMask > 0) - binMaskOnes
            evalTwos = np.count_nonzero(evalMask > 1)

            #if the overlap is too high don't add the new mask
            if(evalTwos == 0 or (evalOnes/evalTwos) > maskOverlap):
                if rainbow == False:
                    compositMask[0] = compositMask[0] + masks[j] * colors[int(labels[j]-1)][0]
                    compositMask[1] = compositMask[1] + masks[j] * colors[int(labels[j]-1)][1] 
                    compositMask[2] = compositMask[2] + masks[j] * colors[int(labels[j]-1)][2]
                else:
                    compositMask[0] = compositMask[0] + masks[j] * R.randrange(0,255)
                    compositMask[1] = compositMask[1] + masks[j] * R.randrange(0,255)
                    compositMask[2] = compositMask[2] + masks[j] * R.randrange(0,255)
                binMask = (evalMask >= 1)
        #add the final assembled mask to the output
        imageList.append(compositMask * 255)
    return imageList

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
def load_images_from_folder(folder):
    images = []
    imageNames = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename),flags=cv2.IMREAD_COLOR+cv2.IMREAD_IGNORE_ORIENTATION)
        if img is not None:
            images.append(img)
            imageNames.append(filename)
    return images,imageNames

#Reads in a runConfig.json from the root directory.
#If it isn't found asks the user for a path to the config json
def readConfig():
    file = None
    try:
        file = open('configs/runConfig.json')
    except Exception:
        flag = True
        while(flag):
            path = input("No runConfig.json found. Enter the path to a config file:")

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




config = readConfig()

#number of classes includes the background
num_classes = config['numberOfClasses']

imageDimentions = config['imageDimentions']

#get the model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = get_model_instance_segmentation(config['modelPath'])
model.to(device)

#load and transform images
imageList,imageNames = load_images_from_folder(config['inputFolderPath'])
transformedImages = TransformImages(imageList,imageDimentions,device)

#run the model on the transformed images
with torch.no_grad():
    predictions = model(transformedImages)

#filter the predicted images
#minScore bb
#iou threash
#mask threash
#mask overlap
filteredImages = filterOutput(predictions,transformedImages, 
    config['filtering']['boundingBoxMinScore'], 
    config['filtering']['iouThreashold'],
    config['filtering']['maskConfidenceThreashold'],
    config['filtering']['maximumMaskOverlap'], imageDimentions,rainbow=config['Rainbow'])

#save the filtered images
SaveImages(filteredImages,imageNames,config['outputFolderPath'])


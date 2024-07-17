import customDataset as cd
import torchvision.tv_tensors
import customDataset as cd
import torch
import torchvision
from torchvision.utils import draw_bounding_boxes, make_grid
import torchvision.transforms.functional as F
import random as R
from ensemble_boxes import *
import json
import cv2
import torchvision.tv_tensors
import numpy as np
from torchvision.utils import draw_bounding_boxes
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from PIL import Image
from torchvision.utils import save_image


mapDict,trainArray,validationArray = cd.splitData('data/processed/rgbPairs.json')
vds = cd.CustomDataset('data/processed/boundingBoxesBackup.csv','data/processed/rgbPairs.json','data/raw/origionalImages/','data/raw/segmentedImages/',trainArray,mapDict,validation=True)

#from the example sets up the model with the number of classes
def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )
    return model


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
#numbere of classes includes the background
num_classes = 5
#get the model
model = get_model_instance_segmentation(num_classes)
#load the model checkpoint
checkpoint = torch.load('model_checkpoints/1024_V2_Final.tar')
model.load_state_dict(checkpoint['model_state_dict'])
#put the model in evaluation mode
model.eval()



#filters the output of the model using ZFTurbo's weight-boxes-fusion
#minScore: minimum score for a bounding box to be concidered
#minIou: how much bounding boxes can overlap
#maskThreash: removes bad masks .3 to .7 works well
#the dimentions of the image width and height must  be the same
def filterOutput(predictions,minScore,minIou,maskThreash,imgDimentions):
    ret = list()
    for i in range(len(predictions)):
        pred = predictions[i]

        img = torch.tensor(images[i],dtype=torch.uint8)
        #removes low confidence masks
        masks = (pred["masks"] > maskThreash).squeeze(1)
        scores = pred['scores']
        boxes = pred['boxes']
        labels = pred['labels']

        boxes = boxes/imgDimentions
 
        boxes, newScores, labels = weighted_boxes_fusion([boxes], [scores], [labels], weights=None, iou_thr=minIou, skip_box_thr=minScore)

        newScoreIndex = 0
        maskList = []
        for i in range(len(scores)):
            if scores[i] == newScores[newScoreIndex]:
                newScoreIndex = newScoreIndex + 1
                maskList.append(masks[i])
            if newScoreIndex == len(newScores):
                break

        #boxes, scores, labels = weighted_boxes_fusion(boxList, scoreList, labelList, weights=None, iou_thr=minIou, skip_box_thr=minScore)
        ret.append([boxes,newScores,labels,maskList])
    return ret

#displays a 4X4 grid of four imges, each with the origional, the training mask,
#predicted boudning boxes, and predicted masks
#filteredList: the filtered output of the model predicitons
#images: the images from the dataloader
#imgDimentions: the dimentions of the images must be the same width and height
#rgbPairsJson: location of the rgbPairsJson file
#the allowable ammount of overlap a new mask can have with those already in the image 1 is 50% 0 is 100%
def showOutput(filteredList,images,imgDimentions,rgbPairsJson,maskOverlap):
    
    rgbDict = json.load(open(rgbPairsJson,'r'))

    #imageList holds all images that will go in the grid
    imageList = list()
    #define colors for showing each class
    colors = [[255,0,0],[0,255,0],[0,0,255],[128,128,0],[0,128,128]]

    for i in range(len(filteredList)):
        #read in the origional image and the associated mask
        maskImage = cv2.imread('data/raw/segmentedImages/' + ids[i] + '.png')
        image = cv2.imread('data/raw/origionalImages/' + rgbDict[ids[i]]['externalID'],flags= cv2.IMREAD_COLOR + cv2.IMREAD_IGNORE_ORIENTATION)
        maskImage = cv2.resize(maskImage, (imgDimentions,imgDimentions))
        image = cv2.resize(image, (imgDimentions,imgDimentions))
        imageList.append(np.transpose(torchvision.tv_tensors.Image(image), axes=(2,0,1)))
        imageList.append(np.transpose(torchvision.tv_tensors.Image(maskImage), axes=(2,0,1)))

        #get the first image from the model
        index = filteredList[i]
        boxes, scores, labels, masks = index
        
        #draw the bounding boxes and add them to the image list
        img = torch.tensor(images[i],dtype=torch.uint8)
        imageList.append(draw_bounding_boxes(img,torch.tensor(boxes*imgDimentions,dtype=torch.int64),width=5))
        
        #The color mask that will be displayed at the end
        compositMask = torch.zeros(3,imgDimentions,imgDimentions)
        #tracks what parts of the mask already have data in them AKA not background
        #this allows for masks to be added in order from most to least conficence
        #excludeing a mask if it overlaps with a higher conficence mask too much
        binMask = np.zeros([imgDimentions,imgDimentions],dtype=np.int8)

        #loop over all the predicted masks
        for i in range(len(masks)):

            #convert to numpy arrays and compare to the existing mask to check for overlap
            npMask = np.array(masks[i])
            binMaskOnes = np.count_nonzero(binMask == 1)
            evalMask = np.add(npMask,binMask,dtype=np.int8)
            evalOnes = np.count_nonzero(evalMask > 0) - binMaskOnes
            evalTwos = np.count_nonzero(evalMask > 1)
            #print('New ones:' +str(evalOnes) + ' New Twos:' + str(evalTwos),)

            #if the overlap is too high don't add the new mask
            if(evalTwos == 0 or (evalOnes/evalTwos) > maskOverlap):

                compositMask[0] = compositMask[0] + masks[i] * colors[int(labels[i]-1)][0]
                compositMask[1] = compositMask[1] + masks[i] * colors[int(labels[i]-1)][1] 
                compositMask[2] = compositMask[2] + masks[i] * colors[int(labels[i]-1)][2]
                binMask = (evalMask >= 1)
        #add the final assembled mask to the output
        imageList.append(compositMask * 255)
    #save images to folders
    i = 0
    while i < len(imageList):
        sublist = [imageList[i],imageList[i+1],imageList[i+2],imageList[i+3]]
        #save_image(make_grid(sublist,nrow=2),('outputs/img'+ids[int(i/4)]+'.png'))#.save('outputs/img.png')
        F.to_pil_image(make_grid(sublist,nrow=2)).save('outputs/img'+ids[int(i/4)]+'.bmp')
        i = i+4


#get images and ids from the dataloader and store them in a list
images = list()
ids = list()
for i in range(40):
    images.append(vds[i][0])
    ids.append(vds[i][1]['imageID'])


#run the model on the imges
with torch.no_grad():
    predictions = model(images)
    print('predict')

    filtered = filterOutput(predictions, .005, .5, .3, 1024)
    showOutput(filtered,images,1024,'data/processed/rgbPairs.json', 1)

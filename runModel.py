import customDataset as cd
import torchvision.tv_tensors
import customDataset as cd
import torch
import torchvision
from torchvision.utils import draw_bounding_boxes, make_grid
import torchvision.transforms.functional as F
from ensemble_boxes import *
import cv2
import torchvision.tv_tensors
import numpy as np
from torchvision.utils import draw_bounding_boxes
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import os
import albumentations as A


#from the example sets up the model with the number of classes
def get_model_instance_segmentation(num_classes,checkpoint):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
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
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    #put the model in evaluation mode
    model.eval()
    return model




#filters the output of the model using ZFTurbo's weight-boxes-fusion
#minScore: minimum score for a bounding box to be concidered
#minIou: how much bounding boxes can overlap
#maskThreash: removes bad masks .3 to .7 works well
#the dimentions of the image width and height must  be the same
def filterOutput(predictions,images,minScore,minIou,maskThreash,maskOverlap,imgDimentions):
    imageList = list()
    colors = [[255,0,0],[0,255,0],[0,0,255],[255,232,0],[0,128,128]]
    for i in range(len(predictions)):
        pred = predictions[i]

 
        #removes low confidence masks
        masks = (pred["masks"] > maskThreash).squeeze(1)
        scores = pred['scores']
        boxes = pred['boxes']
        labels = pred['labels']

        boxes = boxes/imgDimentions
 
        boxes, newScores, labels = weighted_boxes_fusion([boxes], [scores], [labels], weights=None, iou_thr=minIou, skip_box_thr=minScore)

        newScoreIndex = 0
        maskList = []
        for j in range(len(scores)):
            if scores[j] == newScores[newScoreIndex]:
                newScoreIndex = newScoreIndex + 1
                maskList.append(masks[j])
            if newScoreIndex == len(newScores):
                break

        img = torch.tensor(images[i],dtype=torch.uint8)
        # #img = torchvision.tv_tensors.Image(images[i])
        # img = np.transpose(img, axes=(2, 0, 1))
        # print(img.shape)
        imageList.append(draw_bounding_boxes(img,torch.tensor(boxes*imgDimentions,dtype=torch.int64),width=5))
        
        #The color mask that will be displayed at the end
        compositMask = torch.zeros(3,imgDimentions,imgDimentions)
        #tracks what parts of the mask already have data in them AKA not background
        #this allows for masks to be added in order from most to least conficence
        #excludeing a mask if it overlaps with a higher conficence mask too much
        binMask = np.zeros([imgDimentions,imgDimentions],dtype=np.int8)

        #loop over all the predicted masks
        for j in range(len(masks)):

            #convert to numpy arrays and compare to the existing mask to check for overlap
            npMask = np.array(masks[j])
            binMaskOnes = np.count_nonzero(binMask == 1)
            evalMask = np.add(npMask,binMask,dtype=np.int8)
            evalOnes = np.count_nonzero(evalMask > 0) - binMaskOnes
            evalTwos = np.count_nonzero(evalMask > 1)
            #print('New ones:' +str(evalOnes) + ' New Twos:' + str(evalTwos),)

            #if the overlap is too high don't add the new mask
            if(evalTwos == 0 or (evalOnes/evalTwos) > maskOverlap):

                compositMask[0] = compositMask[0] + masks[j] * colors[int(labels[j]-1)][0]
                compositMask[1] = compositMask[1] + masks[j] * colors[int(labels[j]-1)][1] 
                compositMask[2] = compositMask[2] + masks[j] * colors[int(labels[j]-1)][2]
                binMask = (evalMask >= 1)
        #add the final assembled mask to the output
        imageList.append(compositMask * 255)
    return imageList

def TransformImages(images,dimentions):
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
        ret.append(torchImage)
    return ret

def SaveImages(imageList,imageNames,location):
    i = 0
    while i < len(imageList):
        sublist = [imageList[i],imageList[i+1]]
        #save_image(make_grid(sublist,nrow=2),('outputs/img'+ids[int(i/4)]+'.png'))#.save('outputs/img.png')
        F.to_pil_image(make_grid(sublist,nrow=2)).save(location+imageNames[int(i/2)]+'.bmp')
        i = i+2

def load_images_from_folder(folder):
    images = []
    imageNames = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename),flags=cv2.IMREAD_COLOR+cv2.IMREAD_IGNORE_ORIENTATION)
        if img is not None:
            images.append(img)
            imageNames.append(filename)
    return images,imageNames

#numbere of classes includes the background
num_classes = 2

imageDimentions = 512

#get the model
model = get_model_instance_segmentation(num_classes,'model_checkpoints/512_PetioleOnlyV5_Final.tar')

imageList,imageNames = load_images_from_folder('data/raw/run')
transformedImages = TransformImages(imageList,imageDimentions)


with torch.no_grad():
    predictions = model(transformedImages)

filteredImages = filtered = filterOutput(predictions,transformedImages, .005, .5, .4, .001, imageDimentions)

SaveImages(filteredImages,imageNames,'outputs/processed/')
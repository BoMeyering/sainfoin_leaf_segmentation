import torchvision.tv_tensors
import src.datasets as cd
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import src.exampleCode.utils as utils
from src.exampleCode.engine import train_one_epoch, evaluate
import json
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.rpn import RPNHead
from torch import nn

import torchvision.transforms.functional as F
from ensemble_boxes import *
import random as R
import numpy as np
from torchvision.utils import draw_bounding_boxes

from src.utils import *


#get the model
def GetTrainingModel(num_classes):
    # load an instance segmentation model pre-trained on COCO
    #model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
    model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights='COCO_V1')

    #set the anchor box sizes
    # anchor_generator = AnchorGenerator(
    #     sizes=(
    #         (32,), 
    #         (64,), 
    #         (128,), 
    #         (256,), 
    #         (512,),
    #     ),
    #     aspect_ratios=(
    #         (0.25, 0.5, 1.0, 2.0, 3.0, 4.0),
    #         (0.25, 0.5, 1.0, 2.0, 3.0, 4.0),
    #         (0.25, 0.5, 1.0, 2.0, 3.0, 4.0),
    #         (0.25, 0.5, 1.0, 2.0, 3.0, 4.0),
    #         (0.25, 0.5, 1.0, 2.0, 3.0, 4.0),
    #     )
    # )
    # rpn_head = RPNHead(model.backbone.out_channels, anchor_generator.num_anchors_per_location()[0],conv_depth=2)
    # model.rpn.head = rpn_head
    # model.rpn.anchor_generator = anchor_generator


    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    print(in_features)
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    print(in_features_mask)
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )


    return model


def TrainModel():
    config = ReadConfig('configs/trainConfig.json')

    #get the training dataset and the validation dataset
    mapDict,trainArray,validationArray = cd.splitData(config['paths']['rgbJson'])
    tds = cd.CustomDataset(config['paths']['boundingBoxCsv'],
                        config['paths']['rgbJson'],
                        config['paths']['originalImages'],
                        config['paths']['segmentedImages'],
                        trainArray,mapDict)
    vds = cd.CustomDataset(config['paths']['boundingBoxCsv'],
                        config['paths']['rgbJson'],
                        config['paths']['originalImages'],
                        config['paths']['segmentedImages'],
                        validationArray,mapDict,validation=True)



    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # number of classes in the dataset
    num_classes = 3

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        tds,
        batch_size=config['dataLoader']['training']['batchSize'],
        shuffle=config['dataLoader']['training']['shuffle'],
        collate_fn=utils.collate_fn
    )

    data_loader_test = torch.utils.data.DataLoader(
        vds,
        batch_size=config['dataLoader']['validation']['batchSize'],
        shuffle=config['dataLoader']['validation']['shuffle'],
        collate_fn=utils.collate_fn
    )

    # get the model
    model = GetTrainingModel(num_classes)
    #use for pure default model note that the result viewer will also need to be modified
    #model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=config['training']['learningRate'],
        momentum=config['training']['momentum'],
        weight_decay=config['training']['weightDecay']
    )

    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config['training']['stepSize'],
        gamma=config['training']['gamma']
    )

    #the number of epochs to run
    num_epochs = config['training']['numberOfEpochs']

    modelName = config['modelName']

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)
        #save every 4 epochs
        if epoch % config['training']['saveInterval'] == 0:
            torch.save(model,'model_checkpoints/'+modelName+'_'+str(epoch)+'.pt')

    #save the final model
    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }, 'model_checkpoints/1024_'+modelName+'_Final.tar')
    torch.save(model,'model_checkpoints/1024_'+modelName+'_Final.pt')




#Loads a trianed model from a file
def LoadModel(checkpoint,device):

    model = torch.load(checkpoint,map_location=device)
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


def RunModel():

    config = ReadConfig('configs/runConfig.json')

    #number of classes includes the background
    num_classes = config['numberOfClasses']

    imageDimentions = config['imageDimentions']

    #get the model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = LoadModel(config['modelPath'],device)
    model.to(device)

    #load and transform images
    imageList,imageNames = LoadImagesFromFolder(config['inputFolderPath'])
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

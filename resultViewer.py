import matplotlib.pyplot as plt
import exampleCode.utils as utils
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

mapDict,trainArray,validationArray = cd.splitData('data/processed/rgbPairs.json')
vds = cd.CustomDataset('data/processed/boundingBoxesBackup.csv','data/processed/rgbPairs.json','data/raw/origionalImages/','data/raw/segmentedImages/',trainArray,mapDict,validation=True)

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
num_classes = 5
model = get_model_instance_segmentation(num_classes)
checkpoint = torch.load('model_checkpoints/1024_V2_24.tar')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

images = list()
ids = list()
for i in range(4):
    images.append(vds[i][0])
    ids.append(vds[i][1]['imageID'])

with torch.no_grad():
    # convert RGBA -> RGB and move to device
    #i = images.to(device)
    predictions = model(images)
    # pred = predictions[0]


def filterOutput(predictions,minScore,minIou):
    ret = list()
    for i in range(4):
        pred = predictions[i]

        img = torch.tensor(images[i],dtype=torch.uint8)
        #masks = pred['masks']
        masks = (pred["masks"] > 0.3).squeeze(1)
        scores = pred['scores']
        boxes = pred['boxes']
        labels = pred['labels']

        boxes = boxes/1024

        # boxList = list()
        # maskList = list()
        # scoreList = list()
        # labelList = list()
        # for j in range(len(scores)):
        #     if scores[j] > minScore:
        #         scoreList.append(scores[j])
        #         boxList.append(boxes[j])
        #         maskList.append(masks[j])
        #         labelList.append(labels[j])
 
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

def showOutput(filteredList,images):

    rgbDict = json.load(open('data/processed/rgbPairs.json','r'))


    imageList = list()
    colors = [[255,0,0],[0,255,0],[0,0,255],[128,128,0],[0,128,128]]
    #colors = [[255,255,255],[255,255,255],[255,255,255],[255,255,255]]
    for i in range(len(filteredList)):

        maskImage = cv2.imread('data/raw/segmentedImages/' + ids[i] + '.png')
        image = cv2.imread('data/raw/origionalImages/' + rgbDict[ids[i]]['externalID'],flags= cv2.IMREAD_COLOR + cv2.IMREAD_IGNORE_ORIENTATION)
        maskImage = cv2.resize(maskImage, (1024,1024))
        image = cv2.resize(image, (1024,1024))
        imageList.append(np.transpose(torchvision.tv_tensors.Image(image), axes=(2,0,1)))
        imageList.append(np.transpose(torchvision.tv_tensors.Image(maskImage), axes=(2,0,1)))


        index = filteredList[i]
        boxes, scores, labels, masks = index
        


        img = torch.tensor(images[i],dtype=torch.uint8)
        imageList.append(draw_bounding_boxes(img,torch.tensor(boxes*1024,dtype=torch.int64),width=5))
        
        compositMask = torch.zeros(3,1024,1024)
        binMask = np.zeros([1024,1024],dtype=np.int8)
        #print('masks' + str(len(masks)) + ' labels' + str(len(labels)))
        for i in range(len(masks)):

            npMask = np.array(masks[i])
            #print(np.count_nonzero(npMask == 1))
            #uniqueMask, countsMask = np.unique(npMask, return_counts=True)
            binMaskOnes = np.count_nonzero(binMask == 1)
            evalMask = np.add(npMask,binMask,dtype=np.int8)
            evalOnes = np.count_nonzero(evalMask > 0) - binMaskOnes
            evalTwos = np.count_nonzero(evalMask > 1)
            print('New ones:' +str(evalOnes) + ' New Twos:' + str(evalTwos),)
            # if evalTwos == 0:
            #     print('No overlap')
            # else:
            #     print('Overlap percent: ' + str(evalOnes/evalTwos))
  
            if(evalTwos == 0 or (evalOnes/evalTwos) > .001):

                compositMask[0] = compositMask[0] + masks[i] * colors[int(labels[i]-1)][0]
                compositMask[1] = compositMask[1] + masks[i] * colors[int(labels[i]-1)][1] 
                compositMask[2] = compositMask[2] + masks[i] * colors[int(labels[i]-1)][2]
                binMask = (evalMask >= 1)
        imageList.append(compositMask * 255)
    F.to_pil_image(make_grid(imageList,nrow=4)).show()


filtered = filterOutput(predictions, .15, .5)
showOutput(filtered,images)




# imageList = list()
# for i in range(8):
#     pred = predictions[i]
#     img = torch.tensor(images[i],dtype=torch.uint8)
#     result = draw_bounding_boxes(img,pred['boxes'],width=5)
#     imageList.append(result)
#     masks = (pred["masks"] > 0.7).squeeze(1)
#     compositMask = torch.zeros(3,512,512)
#     for m in masks:
#         compositMask[0] = compositMask[0] + m * R.randrange(255)
#         compositMask[1] = compositMask[1] + m * R.randrange(255)
#         compositMask[2] = compositMask[2] + m * R.randrange(255)
#     imageList.append(compositMask)
# F.to_pil_image(make_grid(imageList,nrow=4)).show()
    


# print(pred['scores'])

#image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)
#image = image[:3, ...]
#pred_labels = [f"pedestrian: {score:.3f}" for label, score in zip(pred["labels"], pred["scores"])]
#pred_boxes = pred["boxes"].long()
#output_image = draw_bounding_boxes(image, pred_boxes, pred_labels, colors="red")

# masks = (pred["masks"] > 0.7).squeeze(1)
# #output_image = draw_segmentation_masks(image, masks, alpha=0.5, colors="blue")

# img = torch.tensor(image,dtype=torch.uint8)
# result = draw_bounding_boxes(img,pred['boxes'],width=5)
# F.to_pil_image(result).show()

#plt.figure(figsize=(12, 12))
#plt.imshow(output_image.permute(1, 2, 0))
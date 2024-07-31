 
import torchvision.tv_tensors
import customDataset as cd
import torch
import torchvision
import cv2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import exampleCode.utils as utils
from exampleCode.engine import train_one_epoch, evaluate
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.rpn import RPNHead
from torch import nn



#get the model
def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    #model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
    model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights='COCO_V1')

    print(model.rpn.anchor_generator)

    #set the anchor box sizes
    anchor_generator = AnchorGenerator(
        sizes=(
            (32,), 
            (64,), 
            (128,), 
            (256,), 
            (512,),
        ),
        aspect_ratios=(
            (0.25, 0.5, 1.0, 2.0, 3.0, 4.0),
            (0.25, 0.5, 1.0, 2.0, 3.0, 4.0),
            (0.25, 0.5, 1.0, 2.0, 3.0, 4.0),
            (0.25, 0.5, 1.0, 2.0, 3.0, 4.0),
            (0.25, 0.5, 1.0, 2.0, 3.0, 4.0),
        )
    )
    rpn_head = RPNHead(model.backbone.out_channels, anchor_generator.num_anchors_per_location()[0],conv_depth=2)
    model.rpn.head = rpn_head
    model.rpn.anchor_generator = anchor_generator


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

#get the training dataset and the validation dataset
mapDict,trainArray,validationArray = cd.splitData('data/processed/rgbPairs.json')
tds = cd.CustomDataset('data/processed/boundingBoxesBackup.csv','data/processed/rgbPairs.json','data/raw/origionalImages/','data/raw/segmentedImages/',trainArray,mapDict)
vds = cd.CustomDataset('data/processed/boundingBoxesBackup.csv','data/processed/rgbPairs.json','data/raw/origionalImages/','data/raw/segmentedImages/',validationArray,mapDict,validation=True)



# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#device = torch.device('cpu')

# file = open('model_checkpoints/debug.txt')
# if torch.cuda.is_available:
#     file.write('using GPU')
# else:
#     file.write('using CPU')
# file.close()

# number of classes in the dataset
num_classes = 5

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    tds,
    batch_size=12,
    shuffle=True,
    collate_fn=utils.collate_fn
)

data_loader_test = torch.utils.data.DataLoader(
    vds,
    batch_size=1,
    shuffle=False,
    collate_fn=utils.collate_fn
)

# get the model
model = get_model_instance_segmentation(num_classes)
#use for pure default model note that the result viewer will also need to be modified
#model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

# move model to the right device
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(
    params,
    lr=0.004,
    momentum=0.9,
    weight_decay=0.0005
)

# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=4,
    gamma=0.5
)

#the number of epochs to run
num_epochs = 80

modelName = 'AllClasses'

for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    evaluate(model, data_loader_test, device=device)
    #save every 4 epochs
    if epoch % 5 == 0:
    #     torch.save({
    #         'epoch': epoch,
    #         'model_state_dict': model.state_dict(),
    #         'optimizer_state_dict': optimizer.state_dict()
    #         }, 'model_checkpoints/1024_'+modelName+'_'+str(epoch)+'.tar')
        torch.save(model,'model_checkpoints/1024_'+modelName+'_'+str(epoch)+'.pt')

#save the final model
torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
            }, 'model_checkpoints/1024_'+modelName+'_Final.tar')
torch.save(model,'model_checkpoints/1024_'+modelName+'_Final.pt')



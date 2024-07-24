 
import torchvision.tv_tensors
import customDataset as cd
import torch
import torchvision
import cv2




mapDict,trainArray,validationArray = cd.splitData('data/processed/rgbPairs.json')
tds = cd.CustomDataset('data/processed/boundingBoxesBackup.csv','data/processed/rgbPairs.json','data/raw/origionalImages/','data/raw/segmentedImages/',trainArray,mapDict)
vds = cd.CustomDataset('data/processed/boundingBoxesBackup.csv','data/processed/rgbPairs.json','data/raw/origionalImages/','data/raw/segmentedImages/',validationArray,mapDict)


# model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
# data_loader = torch.utils.data.DataLoader(
#     tds,
#     batch_size=2,
#     shuffle=True,
#     collate_fn=None
# )

# # For Training
# #images, targets = next(iter(data_loader))
# #images = list(image for image in images)
# #targets = [{k: v for k, v in t.items()} for t in targets]
# images, targets = tds[0]
# #print(targets['boxes'])

# #images = cv2.imread('data/raw/segmentedImages/PI313023_001.jpg')

# target = {'boxes': torch.tensor([4,5,6,7])}
# target['masks'] = torchvision.tv_tensors.Image(cv2.imread('data/raw/segmentedImages/cllay52w31scw07a0hfuragq4.png'))
# target['labels'] = torch.tensor([1])
# target['image_id'] = torch.tensor([1])
# target['area'] = torch.tensor([2])
# target['iscrowd'] = torch.tensor([0])


# t = list()
# t.append(targets)

# i = list()
# i.append(images)

# # for target in t:
# #     boxes = target["boxes"]
# #     if isinstance(boxes, torch.Tensor):
# #         torch._assert(
# #             len(boxes.shape) == 2 and boxes.shape[-1] == 4,
# #             f"Expected target boxes to be a tensor of shape [N, 4], got {boxes.shape}.",
# #         )


# output = model(i, t)  # Returns losses and detections
# print(output)

import exampleCode.utils as utils
from exampleCode.engine import train_one_epoch, evaluate


# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# our dataset has two classes only - background and person
num_classes = 4

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    tds,
    batch_size=2,
    shuffle=True,
    collate_fn=utils.collate_fn
)

data_loader_test = torch.utils.data.DataLoader(
    vds,
    batch_size=1,
    shuffle=False,
    collate_fn=utils.collate_fn
)

# get the model using our helper function
#model = utils.get_model_instance_segmentation(num_classes)
model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

# move model to the right device
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(
    params,
    lr=0.001,
    momentum=0.9,
    weight_decay=0.0005
)

# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=3,
    gamma=0.1
)

# let's train it just for 2 epochs
num_epochs = 2

for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    evaluate(model, data_loader_test, device=device)
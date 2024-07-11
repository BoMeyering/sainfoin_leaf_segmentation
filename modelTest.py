 
import torchvision.tv_tensors
import customDataset as cd
import torch
import torchvision
import cv2


mapDict,trainArray,validationArray = cd.splitData('data/processed/rgbPairs.json')
tds = cd.CustomDataset('data/processed/boundingBoxesBackup.csv','data/processed/rgbPairs.json','data/raw/origionalImages/','data/raw/segmentedImages/',trainArray,mapDict)

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
data_loader = torch.utils.data.DataLoader(
    tds,
    batch_size=2,
    shuffle=True,
    collate_fn=None
)

# For Training
#images, targets = next(iter(data_loader))
#images = list(image for image in images)
#targets = [{k: v for k, v in t.items()} for t in targets]
images, targets = tds[0]
#print(targets['boxes'])

images = cv2.imread('data/raw/segmentedImages/PI313023_001.jpg')

target = {'boxes': torch.tensor([4,5,6,7])}
target['masks'] = torchvision.tv_tensors.Image(cv2.imread('data/raw/segmentedImages/cllay52w31scw07a0hfuragq4.png'))
target['labels'] = torch.tensor([1])
target['image_id'] = torch.tensor([1])
target['area'] = torch.tensor([2])
target['iscrowd'] = torch.tensor([0])

print(type(target))
print(type(targets))

targets = list()
targets.append(target)
targets.append(target)



# for target in targets:
#     boxes = target["boxes"]
#     if isinstance(boxes, torch.Tensor):
#         torch._assert(
#             len(boxes.shape) == 2 and boxes.shape[-1] == 4,
#             f"Expected target boxes to be a tensor of shape [N, 4], got {boxes.shape}.",
#         )


output = model(images, targets)  # Returns losses and detections
print(output)
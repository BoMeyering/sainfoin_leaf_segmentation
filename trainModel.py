import customDataset as cd
import torch.utils
import albumentations as A
import random as R



class customDataLoader(torch.utils.data.Dataset):
    def __init__(self,mapDict,splitArray):
        self.dataset = cd.CustomDataset('data/processed/boundingBoxesBackup.csv','data/processed/rgbPairs.json','data/raw/segmentedImages/',splitArray,mapDict)

    def __len__(self):
        return len(self.dataset)

    def getNext(self):
        idx = R.randrange(0,len(self.dataset))
        image, target = self.dataset[idx]
        target_img_size = 512

        transform = A.Compose(
        [
            A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            A.Resize(height=target_img_size, width=target_img_size, p=1),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(),
            A.ChannelShuffle(),
            A.GaussianBlur(),
            A.Affine(translate_percent=(-.10, .10), rotate=(-30, 30), shear=(-15, 15)),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format="pascal_voc", min_area=0, min_visibility=0, label_fields=["labels"]
        ),)

        transformed = transform(
            image=image,
            mask=target['masks'],
            bboxes=target['boxes'],)

        target['boxes'] = transformed['bboxes']
        target['masks'] = transformed['mask']
        image = transformed['image']
        return image, target


mapDict,trainArray,validationArray = cd.splitData('data/processed/rgbPairs.json')
vdl = customDataLoader(mapDict,validationArray)

item = vdl.getNext()

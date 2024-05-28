import cv2
import io
import json
import numpy as np
import tqdm

def processImages(imageSourceFolder, jsonSourceFile, csvSavePath,silent = True):
    rgbDictionary = json.load(open(jsonSourceFile,'r'))

    csv = open(csvSavePath,'w')
    for id in tqdm.tqdm(rgbDictionary):
        image = cv2.imread(imageSourceFolder + id + '.png')
        # imgs = cv2.resize(image,(400,400))
        # cv2.imshow('img',imgs)
        # cv2.waitKey(0)
        for index in rgbDictionary[id]:
            color = rgbDictionary[id][index]['rgb']
            mask = np.zeros((len(image),len(image[0])))

            brgcolor = (color[2],color[1],color[0])

            mask = (image[:, :, 0:3] == brgcolor).all(2) * 255
            npmask = np.array(mask,dtype=np.uint8)
            
            
            # imgs = cv2.resize(npmask,(400,400))
            # cv2.imshow('img',imgs)
            # cv2.waitKey(0)

            contours, hierarchy = cv2.findContours(npmask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2:]
            if len(contours) > 0:
                x,y,w,h = cv2.boundingRect(contours[0])
                x2 = x+w
                y2 = y+h 
                csv.write(f'{id},{index},{x},{y},{x2},{y2}\n')
                if not silent:
                    print(f'X:{x} Y:{y} W:{w} h:{h}')


# image = cv2.imread('data/raw/segmentedImages/cllay52vc1rb407a06rxa4gp6.png')
# print((image[300, 300]))
# mask = (image[:, :, 0:3] == [0,0,128]).all(2) * 255
# npmask = np.array(mask,dtype=np.uint8)
# print((mask[300,300]))
# gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
# print(gray)
# print(mask)
#contours, hierarchy = cv2.findContours(npmask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2:]


processImages('data/raw/segmentedImages/','data/processed/rgbPairs.json','data/processed/boundingBoxes.csv',silent=True)
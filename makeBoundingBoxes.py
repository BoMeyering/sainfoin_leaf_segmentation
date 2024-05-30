import cv2
import io
import json
import numpy as np
import tqdm

#creates a csv in the form imageID,index,x1,y1,x2,y2
#It will save to csvSavePath and requires the rgbjson and rgbimage folder
def processImages(imageSourceFolder, jsonSourceFile, csvSavePath,silent = True):
    #load the rgb dictionary
    rgbDictionary = json.load(open(jsonSourceFile,'r'))
    #create the csv file for writing
    csv = open(csvSavePath,'w')
    #for each image ID
    for id in tqdm.tqdm(rgbDictionary):
        #import the image corosponding to the ID from the dictionary
        image = cv2.imread(imageSourceFolder + id + '.png')
        #for each index AKA annotation
        for index in rgbDictionary[id]:
            #get the color of the anotation
            rgb = rgbDictionary[id][index]['rgb']
            bgr = rgb[::-1]
            bgr = np.array(bgr,dtype=np.uint8)
            # create a black and white mask image that contains only the anotation
            mask = cv2.inRange(image, bgr, bgr)

            # extract the contours form the black and white mask
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # make sure a contour was extracted before accessing it
            if len(cnts) > 0:
                #convert x,y,w,h to x1,y1,x2,y2
                x,y,w,h = cv2.boundingRect(cnts[0])
                x2 = x+w
                y2 = y+h
                #write the line to the csv file
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
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
        image = cv2.imread(imageSourceFolder + id + '.png',flags= cv2.IMREAD_COLOR)
        #for each index AKA annotati on
        for index in rgbDictionary[id]:
            if not isinstance(rgbDictionary[id][index],str):
                #get the color of the anotation
                rgb = rgbDictionary[id][index]['rgb']
                if not silent:
                    print(rgbDictionary[id][index]['class'])
                bgr = rgb[::-1]
                bgr = np.array(bgr,dtype=np.uint8)
                # create a black and white mask image that contains only the anotation
                mask = cv2.inRange(image, bgr, bgr)
                #mask = cv2.inRange(image,(13,9,89),(13,9,89))
                #show_img(mask)

                # extract the contours form the black and white mask
                #cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                #show_img(cv2.drawContours(mask,cnts,-1,(0,255,0),5))
                # make sure a contour was extracted before accessing it
                #if len(cnts) > 0:
                if 1 != 0:
                    #convert x,y,w,h to x1,y1,x2,y2
                    x,y,w,h = cv2.boundingRect(mask)
                    x2 = x+w
                    y2 = y+h
                    #write the line to the csv file
                    csv.write(f'{id},{index},{x},{y},{x2},{y2}\n')
                    if not silent:
                        print(f'X:{x} Y:{y} W:{w} h:{h}')

def show_img(img):
     cv2.namedWindow('test', cv2.WINDOW_NORMAL)
     cv2.imshow('test', img)
     cv2.waitKey()
     cv2.destroyAllWindows()
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
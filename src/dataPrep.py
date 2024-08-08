import labelbox as lb
import json
import requests
import cv2
import json
import numpy as np
import tqdm


# Callback used for JSON Converter
def json_stream_handler(output: lb.JsonConverterOutput, jsonFile, imagePath = 'data/raw/segmentedImages/', silent = False):
  #convert the json to a dictionary
  line = json.loads(output.json_str)
  #check for an error where composite_mask does not appear in the json or the image hasn't been anotated yet
  if len(line['projects']['clixbl663083u07zxhfgxgfio']['labels']) > 0 and 'composite_mask' in line['projects']['clixbl663083u07zxhfgxgfio']['labels'][0]['annotations']['objects'][2]:
    #find the mask url in the json
    url = line['projects']['clixbl663083u07zxhfgxgfio']['labels'][0]['annotations']['objects'][2]['composite_mask']['url']
    #optional print the url
    if not silent:
      print(url)
    #configure the headers for the http request with the API_KEY as authorization
    API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJjbGhpNmJiY2swMGt5MDd3YzlncnA1Z2g4Iiwib3JnYW5pemF0aW9uSWQiOiJjbGhpNmJiYzMwMGt4MDd3Y2Q0aG84cnhoIiwiYXBpS2V5SWQiOiJjbHdqbHFja3YwMDZvMDd4YWVtYWFmb2IyIiwic2VjcmV0IjoiNjNkYWQwNGViYzg2NDViOGYzYTY4ZjMwYTNjNTI4MTQiLCJpYXQiOjE3MTY0ODk4MzksImV4cCI6MjM0NzY0MTgzOX0.DGGptdFGhM8VWn7QyGRQC_JiQvb91KfCPP08zroX9GA"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    #get the image from the url
    webData = requests.get(url,headers=headers).content
    #open a file to write the data to
    file = open(imagePath+(line['data_row']['id']+'.png'), 'wb')
    #write the image to the file
    file.write(webData)
    file.close()
    #write the output to an ndjson file as well
    jsonFile.writelines(output.json_str+'\n')

#retrieves images from label box given a project ID and stores them at imageFolderPath. A ndjson file will be made at jsonPath.
#silent prevents printing to the console
def DownloadImages(projectID,imageFolderPath,jsonPath, silent = False):
  #Connect to the API using the API key
  API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJjbGhpNmJiY2swMGt5MDd3YzlncnA1Z2g4Iiwib3JnYW5pemF0aW9uSWQiOiJjbGhpNmJiYzMwMGt4MDd3Y2Q0aG84cnhoIiwiYXBpS2V5SWQiOiJjbHdqbHFja3YwMDZvMDd4YWVtYWFmb2IyIiwic2VjcmV0IjoiNjNkYWQwNGViYzg2NDViOGYzYTY4ZjMwYTNjNTI4MTQiLCJpYXQiOjE3MTY0ODk4MzksImV4cCI6MjM0NzY0MTgzOX0.DGGptdFGhM8VWn7QyGRQC_JiQvb91KfCPP08zroX9GA"
  client = lb.Client(api_key=API_KEY)

  # get a project using its project ID
  project = client.get_project(projectID) #replace with clixbl663083u07zxhfgxgfio

  # Set the export params to exclude certain fields. 
  # Do not retrieve any extra information
  export_params= {
    "attachments": False,
    "metadata_fields": False,
    "data_row_details": False,
    "project_details": False,
    "label_details": False,
    "performance_details": False,
    "interpolated_frames": False, 
    "embeddings": False
  }

  # Make a filter that should do nothing
  filters= {
    "last_activity_at": ["2000-01-01 00:00:00", "2050-01-01 00:00:00"]
  }

  #export the project
  export_task = project.export(params=export_params, filters=filters)
  export_task.wait_till_done()


  #print erors if export fails
  if export_task.has_errors():
    export_task.get_stream(
    
    converter=lb.JsonConverter(),
    stream_type=lb.StreamType.ERRORS
    ).start(stream_handler=lambda error: print(error))
  #download images if export suceeds
  if export_task.has_result():
    jsonFile = open(jsonPath, mode='w')

    stream = export_task.get_stream()
    for output in stream:
      output: lb.JsonConverterOutput = output
      json_stream_handler(output,jsonFile=jsonFile,imagePath=imageFolderPath,silent=silent)
    jsonFile.close()


  #print the number of lines and file size of the exported json
  if not silent:
    print("file size: ", export_task.get_total_file_size(stream_type=lb.StreamType.RESULT))
    print("line count: ", export_task.get_total_lines(stream_type=lb.StreamType.RESULT))


def MakeImageJson(inputJsonPath, outputJsonPath,silent = False):
  #open the file to write the json to
  jsonFile = open(inputJsonPath,mode='r')

  buildDictionary = {}
  lines = 0
  #assemble a dictionary in the structure
  # ImageID:{
  #   Index:{
  #     RGB Value, Class
  #   }
  # }
  #
  for line in jsonFile:
    lines += 1

    di = json.loads(line)

    rgbDictionary = {}
    #Objects contains all the information about the mask and rgb values
    objects = di['projects']['clixbl663083u07zxhfgxgfio']['labels'][0]['annotations']['objects']
    for i in range(1,len(objects)):
      if 'composite_mask' in objects[i]:
        #extract the rgb value for the mask
        rgb = objects[i]['composite_mask']['color_rgb']
        #extract the class
        cl = objects[i]['value']
        #make them a key value pair in a dictionary
        rgbDictionary[i] = {'rgb':rgb,'class':cl}
      #build the dictionary from the image ID and rgb dictionary  
      rgbDictionary['externalID'] = di['data_row']['external_id']
      buildDictionary[di['data_row']['id']] = rgbDictionary

  #open a file to write to
  outFile = open(outputJsonPath,mode='w')
  #write the full dictionary to the file
  json.dump(buildDictionary,outFile,indent=4)
  outFile.close()
  if not silent:
    print(f'wrote {lines} lines to data/processed/rgbPairs.json')


#creates a csv in the form imageID,index,x1,y1,x2,y2
#It will save to csvSavePath and requires the rgbjson and rgbimage folder
def makeBoundingBoxes(imageSourceFolder, jsonSourceFile, csvSavePath,silent = True):
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
                #extract bounding rectanges 
                x,y,w,h = cv2.boundingRect(mask)
                #convert x,y,w,h to x1,y1,x2,y2
                x2 = x+w
                y2 = y+h
                #write the line to the csv file
                csv.write(f'{id},{index},{x},{y},{x2},{y2}\n')
                if not silent:
                    print(f'X:{x} Y:{y} W:{w} h:{h}')
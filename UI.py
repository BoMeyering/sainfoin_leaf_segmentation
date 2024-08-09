from src.models import TrainModel, RunModel
from src.dataPrep import DownloadImages, MakeImageJson, makeBoundingBoxes

def getCommands():
    numCommands = []
    validInput = False
    while validInput == False:
        i = input('''Choose an option enter multiple commands seperated with commas:
    0) Download images from Labelbox
    1) Make the rgbJson file
    2) Make the bounding box csv file
    3) Train a model
    4) Run a model
    5) exit\n''')

        commands = i.split(',')

        for j in commands:
            try:
                numCommands.append(int(j))
                validInput = True
            except:
                print("invalid input")
                validInput = False
                break
    return numCommands

while True:
    numCommands = getCommands()
    for cmd in numCommands:
        if cmd == 0:
            DownloadImages(projectID='clixbl663083u07zxhfgxgfio', imageFolderPath='data/raw/segmentedImages/', jsonPath='data/raw/exportProject.ndjson')
        elif cmd == 1:
            MakeImageJson(inputJsonPath='data/raw/exportProject.ndjson', outputJsonPath='data/processed/rgbPairs.json')
        elif cmd == 2:
            makeBoundingBoxes('data/raw/segmentedImages/','data/processed/rgbPairs.json','data/processed/boundingBoxes.csv',silent=True)
        elif cmd == 3:
            TrainModel()
        elif cmd == 4:
            RunModel()
        elif cmd == 5:
            exit()


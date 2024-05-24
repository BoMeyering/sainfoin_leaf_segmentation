import urllib.request
from PIL import Image
import json
from io import StringIO



jsonFile = open('data/raw/exportProject.ndjson',mode='r')

buildDictionary = {}
lines = 0

for line in jsonFile:
    lines += 1

    di = json.loads(line)

    rgbDictionary = {}
    objects = di['projects']['clixbl663083u07zxhfgxgfio']['labels'][0]['annotations']['objects']
    for i in range(1,len(objects)):
        if 'composite_mask' in objects[i]:
            rgb = objects[i]['composite_mask']['color_rgb']
            cl = objects[i]['value']
            rgbDictionary[i] = {'rgb':rgb,'class':cl}
            
    buildDictionary[di['data_row']['id']] = rgbDictionary


outFile = open('data/processed/rgbPairs.json',mode='w')
json.dump(buildDictionary,outFile,indent=4)
outFile.close()
print(f'wrote {lines} lines to data/processed/rgbPairs.json')
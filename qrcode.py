import cv2
import os
from PIL import Image
from nicegui import ui,app




def LoadQRImages(folder):
    images = []
    qrcodes = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename),flags=cv2.IMREAD_COLOR+cv2.IMREAD_IGNORE_ORIENTATION)
        if img is not None:
            qrcodes.append(img[0:768,0:1024])
            images.append(img)
    return images,qrcodes



#img,qrcodes = LoadQRImages('qrDetection/input')

detector = cv2.QRCodeDetector()

# folder = 'qrDetection/input'
# for filename in os.listdir(folder):
#     print(filename)
#     img = cv2.imread(os.path.join(folder,filename),flags=cv2.IMREAD_COLOR+cv2.IMREAD_IGNORE_ORIENTATION)
#     if img is not None or True:
#         qrcode = (img[0:768,0:1024])
#         retval, decoded_info, points, straight_qrcode = detector.detectAndDecodeMulti(qrcode)
#         if(retval):
#             appendString = ''
#             i = 0
#             print(os.path.isfile('qrDetection/output/' + decoded_info[0] + appendString + '.jpg'))
#             while os.path.isfile('qrDetection/output/' + decoded_info[0] + appendString + '.jpg'):
#                 appendString = str(i)
#                 i = i + 1

#             cv2.imwrite('qrDetection/output/' + decoded_info[0] + appendString + '.jpg',img)
#             os.remove(os.path.join(folder,filename))

# ui.image('https://picsum.photos/id/377/640/360')
# x = cv2.imread('qrDetection/input/TLI_E+K_001.jpg')
# displayImage = ui.image(x)
# displayImage.style('width: 1024px; height: 768px')
# button = ui.button('Next')

class textString():
    def __init__(self):
        self.text = ''
    def getText(self):
        return self.text
    


folder = 'qrDetection/input'
displayImage = ui.image('https://picsum.photos/id/377/640/360')
displayImage.style('width: 1024px; height: 768px')
ts = textString()
textBox = ui.input(label='Text', placeholder='start typing',
        validation={'Input too long': lambda value: len(value) < 20})
textBox.bind_value(ts,'text')



@ui.page('/')
async def index():
    folder = 'qrDetection/input'
    displayImage = ui.image('https://picsum.photos/id/377/640/360')
    displayImage.style('width: 1024px; height: 768px')
    ts = textString()
    textBox = ui.input(label='Text', placeholder='start typing',
         validation={'Input too long': lambda value: len(value) < 20})
    textBox.bind_value(ts,'text')
    
    b = ui.button('Step')

    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename),flags=cv2.IMREAD_COLOR+cv2.IMREAD_IGNORE_ORIENTATION)
        if img is not None:
            qrcode = (img[0:768,0:1024])
            retval, decoded_info, points, straight_qrcode = detector.detectAndDecodeMulti(qrcode)
            if(retval):
                #displayImage.set_source(Image.fromarray(qrcode))
                #displayImage.update()
                
                appendString = '_0'
                i = 1
                while os.path.isfile('qrDetection/output/' + decoded_info[0] + appendString + '.jpg'):
                    appendString = '_'+str(i)
                    i = i + 1

                cv2.imwrite('qrDetection/output/' + decoded_info[0] + appendString + '.jpg',img)
                #ui.notify("Found QR: " + 'qrDetection/output/' + decoded_info[0] + appendString + '.jpg')
                #ui.update()
                #await b.clicked()
            else:
                displayImage.set_source(Image.fromarray(cv2.rotate(qrcode,cv2.ROTATE_180)))
                displayImage.update()
                ui.notify("No QR found")
                await b.clicked()
                appendString = '_0'
                i = 1
                while os.path.isfile('qrDetection/output/' + ts.getText() + appendString + '.jpg'):
                    appendString = str(i)
                    i = i + 1
                ui.notify("Image saved with name: " + ts.getText())
                cv2.imwrite('qrDetection/output/' + ts.getText() + appendString + '.jpg',img)

    displayImage.set_source('https://picsum.photos/id/377/640/360')
    displayImage.update()
            

ui.run(reconnect_timeout=10,reload=True)
    # for i in range(len(qrcodes)):
    #     retval, decoded_info, points, straight_qrcode = detector.detectAndDecodeMulti(qrcodes[i])
    #     if(retval):
    #         cv2.imwrite('qrDetection/output/' + decoded_info[0] + '.jpg',img[i])
    #     else:
    #         x = Image.fromarray(qrcodes[i])
    #         displayImage.set_source(x)
    #         displayImage.update()
    #         await button.clicked()


#ui.run()#native=True,window_size=(800,600),fullscreen=False)
# async def imageUI():
    # await button.clicked()
    # displayImage.set_source(Image.fromarray(qrcodes[1]))
    # displayImage.update()
    # for i in range(len(qrcodes)):
    #     retval, decoded_info, points, straight_qrcode = detector.detectAndDecodeMulti(qrcodes[i])
    #     if(retval):
    #         cv2.imwrite('qrDetection/output/' + decoded_info[0] + '.jpg',img[i])
    #     else:
    #         x = Image.fromarray(qrcodes[i])
    #         displayImage.set_source(x)
    #         displayImage.update()
    #         await button.clicked()



import cv2
import os




def load_images_from_folder(folder):
    images = []
    qrcodes = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename),flags=cv2.IMREAD_COLOR+cv2.IMREAD_IGNORE_ORIENTATION)
        if img is not None:
            qrcodes.append(img[0:768,0:1024])
            images.append(img)
    return images,qrcodes

img,qrcodes = load_images_from_folder('data/raw/Outliers')

detector = cv2.QRCodeDetector()

for i in range(len(qrcodes)):
    retval, decoded_info, points, straight_qrcode = detector.detectAndDecodeMulti(qrcodes[i])
    if(retval):
        cv2.imwrite('data/processed/' + decoded_info[0] + '.jpg',img[i])
    else:
        cv2.imshow('No QR found',qrcodes[i])
        cv2.waitKey(0)
        name = input("Enter image name:")
        cv2.destroyAllWindows()
        cv2.imwrite('data/processed/' + name + '.jpg',img[i])
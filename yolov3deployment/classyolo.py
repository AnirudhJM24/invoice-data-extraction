
import numpy as np

import time
from code import Lables
import cv2
import os
import io
from PIL import Image
from pdf2image import convert_from_path


pdf_path = '' #pdf input file path
yolo_path = './' #create yolo folder with .names , .cfg and .weights file
path = r'' #images from pdf to be stored here
path2w = r''#extracted and cropped images to be stored here
labelsPath=" "#labels path
cfgpath=" "# config file path
wpath=" " #weight file path

#gets lables
def get_lables(label_path):
    lpath=os.path.sep.join([yolo_path, label_path])
    LABELS = open(lpath).read().strip().split("\n")
    return LABELS
# gets weights
def get_weights(weights_path):
       weightsPath = os.path.sep.join([yolo_path, weights_path])
       return weightsPath
#gets configuration file
def get_config(config_path):
       configPath = os.path.sep.join([yolo_path, config_path])
       return configPath

#yolo model
class model():

    confthres = 0.3
    nmsthres = 0.1

    def __init__(self,weights_path,config_path,lables_path):
        self.weightspath = get_weights(weights_path)
        self.configpath = get_config(config_path)
        self.lables = get_lables(lables_path)

    def set_confthres(self,conf):
        model.confthres = conf
    
    def set_nmsthres(self,nms):
        model.nmsthres = nms
    
    
    def load_model(self):
        print("loading yolo")

        net = cv2.dnn.readNetFromDarknet(self.configpath,self.weightspath)

        return net

    def make_prediction(self,image):

        net = model.load_model()

        (H,W) = image.shape[:2]

        ln = net.getLayerNames()

        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)

        net.setInput(blob)
        start = time.time()
        layerOutputs = net.forward(ln)
        print(layerOutputs)
        end = time.time()

        print("[INFO] YOLO took {:.6f} seconds".format(end - start))

        boxes = []
        confidences = []
        classIDs = []

        for output in layerOutputs:

            for detection in output:

                scores = detection[5:]

                classID = np.argmax(scores)

                confidence = scores[classID]

                if confidence > model.confthres:

                    box = detection[0:4] * np.array([W,H,W,H])

                    (centreX,centreY,width,height) = box.astype('int')

                    x = int(centreX-(width/2))

                    y = int(centreY-(height/2))

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)


        idxs = cv2.dnn.NMSBoxes(boxes, confidences, model.confthres,
                            model.nmsthres)

        if len(idxs)>0:

            for i in idxs.flatten():

                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                c_i = image[y:y+h , x:x+w]               
                cv2.imwrite(os.path.join(path2w,self.lables[classIDs[i]]+'.jpg'),c_i)

#class for preprocessing images
class image_preprocess():
    def __init__(self,image):
        self.image = image
    def process(self):
        pass
        #write code for image processing

# class for pdf to image    
class pdf_to_image():

    def __init__(self):
        pass

    def pdf2img(pdf,path):

        images = convert_from_path('example.pdf')
 
        for i in range(len(images)):

      # Save pages as images in the pdf
            cv2.imwrite(os.path.join(path,str(i)+'.jpg'),images[i])


        return len(images)
       
# class for running OCR

class OCR:
    tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract' #set path for pytesseract installed

    def __init__():
        pass
    def set_path(self,path =r'C:\Program Files\Tesseract-OCR\tesseract' ):
        OCR.tesseract_path = path

    def run_ocr(self,result_path):
        pass



def main():
    pass

if __name__ == '__main__':
    main()






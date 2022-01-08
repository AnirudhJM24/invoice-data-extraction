# -*- coding: utf-8 -*-
"""
Created on Sat Jan  8 10:59:28 2022

@author: anirudh
"""
import numpy as np

import time
import cv2
import os
import io
from PIL import Image
import pytesseract
from pytesseract import Output


confthres = 0.3
nmsthres = 0.1
yolo_path = './'
path = r'results'
path2w = r'results\results.txt'
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract' #replace with where its installed for you XD
a=[]



def run_ocr_custom():
    for fn in os.listdir(path) :
        if fn.endswith('.jpg') and fn != 'TABLE.jpg':           
            x = pytesseract.image_to_string(Image.open(os.path.join(path,fn)),lang='eng')
            a.append(fn[0:len(fn)-3]+':'+x)


def get_labels(labels_path):
    lpath=os.path.sep.join([yolo_path, labels_path])
    LABELS = open(lpath).read().strip().split("\n")
    return LABELS


def get_weights(weights_path):
    weightsPath = os.path.sep.join([yolo_path, weights_path])
    return weightsPath

def get_config(config_path):
    configPath = os.path.sep.join([yolo_path, config_path])
    return configPath

def load_model(configpath,weightspath):
    print("loading yolo")
    net = cv2.dnn.readNetFromDarknet(configpath, weightspath)
    return net





def get_predection(image,net,LABELS):
    (H, W) = image.shape[:2]

    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]


    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    print(layerOutputs)
    end = time.time()

    # show timing information on YOLO
    print("[INFO] YOLO took {:.6f} seconds".format(end - start))

    
    boxes = []
    confidences = []
    classIDs = []


    for output in layerOutputs:
 
        for detection in output:
            
            scores = detection[5:]
            
            classID = np.argmax(scores)
          
            confidence = scores[classID]

            
            if confidence > confthres:
                #
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confthres,
                            nmsthres)

    
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            c_i = image[y:y+h , x:x+w]               
            cv2.imwrite(os.path.join(path,LABELS[classIDs[i]]+'.jpg'),c_i)



        
                 
labelsPath="yolo/classes.names"
cfgpath="yolo/yolov3_custom.cfg"
wpath="yolo/yolov3_custom_last.weights"
Lables=get_labels(labelsPath)
CFG=get_config(cfgpath)
Weights=get_weights(wpath)
nets=load_model(CFG,Weights)


def main():
    image = cv2.imread("./13.jpg")
    npimg=np.array(image)
    image=npimg.copy()
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    get_predection(image,nets,Lables)
    run_ocr_custom()
    f_1 = open(path2w,'w')
    for x in a:
        f_1.writelines(x+'\n')
    

        
if __name__ == '__main__':
    main()

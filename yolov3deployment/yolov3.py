# -*- coding: utf-8 -*-
"""
Created on Sat Jan  8 10:59:28 2022

@author: anirudh
"""
import numpy as np
import sys
import time
import cv2
import os
import io
from PIL import Image
import pytesseract
from pytesseract import Output
from pdf2image import convert_from_path


confthres = 0.3
nmsthres = 0.1
yolo_path = './'
if len(sys.argv) != 2:
    sys.exit("Usage: python yolov3.py <invoice_name>")

# example=os.path.join('examples', sys.argv[1])
# print(example)
# path = os.path.join(os.path.splitext(example)[0])
# print(path)
# os.makedirs(path)
# path2w = os.path.join(path, 'results.txt')
# print(path2w)
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract' #replace with where its installed for you
a=[]



def run_ocr_custom(path):
    for fn in os.listdir(path) :
        if fn.endswith('.jpg') and fn != 'TABLE.jpg':           
            x = pytesseract.image_to_string(Image.open(os.path.join(path,fn)),lang='eng')
            a.append(fn[0:len(fn)-3]+':'+x)

def run_table_ocr(path):
    def pre_process_image(img, save_in_file, morph_size=(9, 12)):
        # get rid of the color
        pre = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Otsu threshold
        pre = cv2.threshold(pre, 250, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        # dilate the text to make it solid spot
        cpy = pre.copy()
        struct = cv2.getStructuringElement(cv2.MORPH_RECT, morph_size)
        cpy = cv2.dilate(~cpy, struct, anchor=(-1, -1), iterations=1)
        pre = ~cpy


        if save_in_file is not None:
            cv2.imwrite(save_in_file, pre)
        return pre


    def find_text_boxes(pre, min_text_height_limit=6, max_text_height_limit=40):
        # Looking for the text spots contours
        
        contours, hierarchy = cv2.findContours(pre, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # Getting the texts bounding boxes based on the text size assumptions
        boxes = []
        for contour in contours:
            box = cv2.boundingRect(contour)
            h = box[3]

            if min_text_height_limit < h < max_text_height_limit:
                boxes.append(box)

        return boxes


    def find_table_in_boxes(boxes, cell_threshold=5, min_columns=2):
        rows = {}
        cols = {}

        # Clustering the bounding boxes by their positions
        for box in boxes:
            (x, y, w, h) = box
            col_key = x // cell_threshold
            row_key = y // cell_threshold
            cols[row_key] = [box] if col_key not in cols else cols[col_key] + [box]
            rows[row_key] = [box] if row_key not in rows else rows[row_key] + [box]

        # Filtering out the clusters having less than 2 cols
        table_cells = list(filter(lambda r: len(r) >= min_columns, rows.values()))
        # Sorting the row cells by x coord
        table_cells = [list(sorted(tb)) for tb in table_cells]
        # Sorting rows by the y coord
        table_cells = list(sorted(table_cells, key=lambda r: r[0][1]))

        return table_cells


    def build_lines(table_cells):
        if table_cells is None or len(table_cells) <= 0:
            return [], []

        max_last_col_width_row = max(table_cells, key=lambda b: b[-1][2])
        max_x = max_last_col_width_row[-1][0] + max_last_col_width_row[-1][2]

        max_last_row_height_box = max(table_cells[-1], key=lambda b: b[3])
        max_y = max_last_row_height_box[1] + max_last_row_height_box[3]

        hor_lines = []
        ver_lines = []

        for box in table_cells:
            x = box[0][0]
            y = box[0][1]
            hor_lines.append((x, y, max_x, y))

        for box in table_cells[0]:
            x = box[0]
            y = box[1]
            ver_lines.append((x, y, x, max_y))

        (x, y, w, h) = table_cells[0][-1]
        ver_lines.append((max_x, y, max_x, max_y))
        (x, y, w, h) = table_cells[0][0]
        hor_lines.append((x, max_y, max_x, max_y))

        return hor_lines, ver_lines


    in_file = os.path.join(path,'TABLE.jpg')
    pre_file = os.path.join(path, "pre.png")
    out_file = os.path.join(path, "out.png")


    if(os.path.exists(in_file)):
        img = cv2.imread(os.path.join(in_file))
    
        pre_processed = pre_process_image(img, pre_file)
        text_boxes = find_text_boxes(pre_processed)
        cells = find_table_in_boxes(text_boxes)
        hor_lines, ver_lines = build_lines(cells)

        # Visualize the result
        vis = img.copy()

        # for box in text_boxes:
        #     (x, y, w, h) = box
        #     cv2.rectangle(vis, (x, y), (x + w - 2, y + h - 2), (0, 255, 0), 1)

        for line in hor_lines:
            [x1, y1, x2, y2] = line
            cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255), 1)

        for line in ver_lines:
            [x1, y1, x2, y2] = line
            cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255), 1)

        # cv2.imshow("out_file", vis)
        # cv2.waitKey(0)
        # cv2.destroyWindow("with_line")
        cv2.imwrite(os.path.join(path,'out.jpg'),vis)



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





def get_predection(image,net,LABELS,path):
    (H, W) = image.shape[:2]

    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]


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
    example=os.path.join('examples', sys.argv[1])
    
    

    if example.endswith('.jpg') or example.endswith('.png'):
        path = os.path.join(os.path.splitext(example)[0])

        os.makedirs(path)
        path2w = os.path.join(path, 'results.txt')

        image = cv2.imread(example)
        npimg=np.array(image)
        image=npimg.copy()
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        get_predection(image,nets,Lables,path)
        run_ocr_custom(path)
        run_table_ocr(path)
        f_1 = open(path2w,'w')
        for x in a:
            f_1.writelines(x+'\n')
    elif example.endswith('.pdf'):
        images = convert_from_path(example)
        for i,image in enumerate(images):

            path = os.path.join(os.path.splitext(example)[0],str(i))

            os.makedirs(path)
            path2w = os.path.join(path, 'results.txt')

            # image = cv2.imread(img)
            npimg=np.array(image)
            image=npimg.copy()
            image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            get_predection(image,nets,Lables,path)
            run_ocr_custom(path)
            run_table_ocr(path)
            f_1 = open(path2w,'w')
            for x in a:
                f_1.writelines(x+'\n')

    

        
if __name__ == '__main__':
    main()


import cv2
import torch
import pandas
from pdf2image import convert_from_path
import os
import shutil
import sys
import pytesseract
from PIL import Image
import argparse

class text_extract:
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    text_write_path = 'results.txt'
    table_write_path = 'table.csv'

    def retrieve_text(self, field_name, image_path):
        #print(field_name,image_path)
        try:
            img = Image.open(image_path)
            img_text = pytesseract.image_to_string(img)
            print(field_name + ':' + img_text)
            with open(self.text_write_path,'a') as f:
                f.writelines(field_name + ':' + img_text)
                f.close()
        except Exception as e:
            #print(e)
            print(field_name + ' not found')
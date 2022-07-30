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

from preprocess import preimgpdf
from yolov import model
from dataextraction import text_extract


parser = argparse.ArgumentParser(description='run invoice detection')
parser.add_argument('--file',help = 'enter path of invoice')
args = parser.parse_args()


def main():
    preprocess = preimgpdf()
    detector = model('weights/best.pt')
    ocr = text_extract()
    invoice_path = args.file
    invoice = invoice_path
      # path to image
    img_supp_types = '.jpg' or '.png'

    if invoice.endswith(img_supp_types):
        detector.predict( invoice_path)
        crop_img_paths, table_dir = preprocess.set_dir(invoice)
        for img in crop_img_paths:
            ocr.retrieve_text(img['field'],img['path'])
        print('TABLE DETAILS:')
        table_path = table_dir
        os.system('python table-extraction\\table_transformer.py --table-type borderless -i ' + table_path)
        #print(crop_img_paths, table_dir)


    elif invoice.endswith('.pdf'):
        images = preprocess.pdf_images(invoice)
        for imag in images:
            detector.predict(imag)
            crop_img_paths, table_dir = preprocess.set_dir(imag)
            for img in crop_img_paths:
                ocr.retrieve_text(img['field'],img['path'])
                try:
                    os.system('python table-extraction\\table_transformer.py --table-type borderless -i ' + table_path)
                except:
                    print("no table")


if __name__ == "__main__":
    main()

# !python3 table_transformer.py --table-type borderless -i "/content/open-intelligence-backend/datasets/all_tables/2.png"
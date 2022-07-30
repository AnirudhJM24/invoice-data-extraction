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

class preimgpdf:
    save_dir = 'runs/detect'
    poppler_path = 'poppler-22.04.0/Library/bin'
    save_path = '/temp'
    def __init__(self):
        print("--------------------")
        print("clearing directory")
        self.clear_directory()

    def clear_directory(self):
        try:
            for f in os.listdir(self.save_dir):
                shutil.rmtree(os.path.join(self.save_dir,f))
            for f in os.listdir(self.save_path):
                shutil.rmtree(os.path.join(self.save_path,f))
            print("Cleared save directory")
        except :
            print('this path does not exist yet')

    def set_dir(self,invoice_path):
        invoice_name  = invoice_path.split('\\')[-1]
        company_dir = os.path.join('runs/detect/exp/crops/COMPANY', invoice_name)
        invoice_date_dir = os.path.join('runs/detect/exp/crops/INVOICE DATE' , invoice_name)
        table_dir = os.path.join('runs/detect/exp/crops/TABLE', invoice_name)
        total_dir = os.path.join('runs/detect/exp/crops/TOTAL' , invoice_name)
        gst_dir = os.path.join('runs/detect/exp/crops/GST' , invoice_name)
        abn_dir = os.path.join('runs/detect/exp/crops/ABN' , invoice_name)
        account_dir = os.path.join('runs/detect/exp/crops/ACCOUNT_DETAILS' , invoice_name)
        crop_img_paths = [{'field' : 'company', 'path' :company_dir} , {'field' : 'invoice date', 'path' : invoice_date_dir} , {'field' : 'Total', 'path' :total_dir}, {'field' : 'gst', 'path' :gst_dir}, {'field' : 'abn', 'path' :abn_dir},{'field' : 'Account Details', 'path' :account_dir}]
        return crop_img_paths, table_dir

    def pdf_images (self,pdfpath):
        pages = convert_from_path(pdfpath,700)
        for i in range(0,len(pages)):
            pages[i].save(os.path.join(self.save_path,i.jpg), 'JPEG')
        paths = []
        for image in os.scandir(self.save_path):
            imgpath = image.path
            imgname = image.name
            paths.append(imgpath)
        return paths



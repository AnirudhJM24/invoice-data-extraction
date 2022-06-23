import cv2
import torch
import pandas
from pdf2image import convert_from_path
import os
import shutil
import sys
import pytesseract
from PIL import Image



pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

with open('results.txt','w') as f:
            f.truncate(0)

save_dir = 'runs/detect'

invoice_path = 'examples\\treebo.jpg'
invoice_name  = invoice_path.split('\\')[-1]

company_dir = os.path.join('runs\detect\exp\crops\COMPANY', invoice_name)
invoice_date_dir = os.path.join('runs\detect\exp\crops\INVOICE DATE' , invoice_name)
table_dir = os.path.join('runs\detect\exp\crops\TABLE', invoice_name)
total_dir = os.path.join('runs\detect\exp\crops\TOTAL' , invoice_name)
gst_dir = os.path.join('runs\detect\exp\crops\GST' , invoice_name)
abn_dir = os.path.join('runs\detect\exp\crops\ABN' , invoice_name)
account_dir = os.path.join('runs\detect\exp\crops\ACCOUNT_DETAILS' , invoice_name)

crop_img_paths = [{'field' : 'company', 'path' :company_dir} , {'field' : 'invoice date', 'path' : invoice_date_dir} , {'field' : 'Total', 'path' :total_dir}, {'field' : 'gst', 'path' :gst_dir}, {'field' : 'abn', 'path' :abn_dir},{'field' : 'Account Details', 'path' :account_dir}]

# os.chmod(save_dir, 0o777)

def clear_directory(save_dir):
    for f in os.listdir(save_dir):
        shutil.rmtree(os.path.join(save_dir,f))


def predict(model, image):
    results = model(image)
    crops = results.crop()
    #print(results.pandas().xyxy[0])

def retrieve_text(field_name, image_path):
    try:
        img = Image.open(image_path)
        img_text = pytesseract.image_to_string(img)
        print(field_name + ':' + img_text)
        with open('results.txt','a') as f:
            f.writelines(field_name + ':' + img_text)
            f.close()
    except:
        print(field_name + ' not found')

def main():
    clear_directory(save_dir)
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='weights/best.pt')
    invoice = 'examples\sample2_invoice3.jpg'  # path to image
    img_supp_types = '.jpg' or '.png'
    if invoice.endswith(img_supp_types):
        predict(model, invoice)
    elif invoice.endswith('pdf'):
        images = convert_from_path(invoice)
        for i in enumerate(images):
            clear_directory(save_dir)
            predict(i)
            for img in crop_img_paths:
                retrieve_text(img['field'],img['path'])
                print('TABLE DETAILS:')
                table_path = table_dir
                os.system('python table-extraction\\table_transformer.py --table-type borderless -i ' + table_path)
                

    
    for img in crop_img_paths:
        retrieve_text(img['field'],img['path'])
    print('TABLE DETAILS:')
    table_path = table_dir
    os.system('python table-extraction\\table_transformer.py --table-type borderless -i ' + table_path)


if __name__ == "__main__":
    main()

# !python3 table_transformer.py --table-type borderless -i "/content/open-intelligence-backend/datasets/all_tables/2.png"
import cv2
import torch
import pandas
from pdf2image import convert_from_path
import os
import shutil

save_dir = 'runs/detect'

os.chmod(save_dir, 0o777)


for f in os.listdir(save_dir):
    shutil.rmtree(os.path.join(save_dir,f))


def predict(image):
    results = model(image)
    crops = results.crop()
    print(results.pandas().xyxy[0])


model = torch.hub.load('ultralytics/yolov5', 'custom', path='weights/best.pt')
invoice = 'examples/sample2_invoice1.jpg'  # path to image
img_supp_types = '.jpg' or '.png'
if invoice.endswith(img_supp_types):
    predict(invoice)
elif invoice.endswith('pdf'):
    images = convert_from_path(invoice)
    for i in enumerate(images):
        predict(i)



import cv2
import torch
import pandas

model = torch.hub.load('ultralytics/yolov5', 'custom', path='weights/best.pt')
results = model('examples/sample2_invoice1.jpg')
crops = results.crop(save=True)

results.pandas().xyxy[0]

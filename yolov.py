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


class model:

    def __init__(self,model_path):
        self.model = torch.hub.load('ultralytics/yolov5','custom', path = model_path)
    
    def predict(self, image):
        results = self.model(image)
        results.crop()


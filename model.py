import cv2
import torch


class GetModel:

    def __init__(self, weights_path):
        print("loading model........")
        self.weights_path = weights_path
        self.load_model()

    def load_model(self):
        detector = torch.hub.load('ultralytics/yolov5', 'custom', path=self.weights_path)
        return detector
    # def predict(self,image):


model = torch.hub.load('ultralytics/yolov5', 'custom', path='weights/best.pt')
results = model('examples/14.jpg')
crops = results.crop(save=True)

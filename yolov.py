import torch


class model:

    def __init__(self, model_path):
        self.model = torch.hub.load(
            'ultralytics/yolov5', 'custom', path=model_path)

    def predict(self, image):
        results = self.model(image)
        results.crop()

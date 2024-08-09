from ultralytics import YOLO
import cv2
from PIL import Image


model = YOLO("LogoYolo/LogoYolobest.pt")
dict_classes = model.model.names

def predict(src):
    result_predict = model.predict(source=src, save=False)  # Set save=False to not save the output
    plot = result_predict[0].plot()
    plot = cv2.cvtColor(plot, cv2.COLOR_BGR2RGB)
    return Image.fromarray(plot)
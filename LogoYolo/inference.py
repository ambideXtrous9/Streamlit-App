from ultralytics import YOLO
from PIL import Image


model = YOLO("LogoYolo/LogoYolobest.pt")
dict_classes = model.model.names

def predict(src):
    result_predict = model.predict(source=src, save=False)  # Set save=False to not save the output
    plot = result_predict[0].plot()
    return Image.fromarray(plot)
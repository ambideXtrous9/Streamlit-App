import os
import torch
from torchvision import transforms
import pytorch_lightning as pl
from PIL import Image
import config
import time
from EfficientNetB0 import EfficientNet
from Xception import XceptionNet
from MobilenetV2 import MobileNetV2
from InceptionV3 import InceptionV3
import warnings
warnings.filterwarnings("ignore")


resize = transforms.Resize(size=(config.WIDTH,config.HEIGHT))

nrml = transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])  # Imagenet standards

transform_norm = transforms.Compose([
            transforms.ToTensor(),
            resize,
            nrml])

# Define the classes dictionary
class_labels = {
    'Adidas': 0, 'Apple': 1, 'BMW': 2, 'Citroen': 3, 'Cocacola': 4, 
    'DHL': 5, 'Fedex': 6, 'Ferrari': 7, 'Ford': 8, 'Google': 9, 
    'HP': 10, 'Heineken': 11, 'Intel': 12, 'McDonalds': 13, 'Mini': 14, 
    'Nbc': 15, 'Nike': 16, 'Pepsi': 17, 'Porsche': 18, 'Puma': 19, 
    'RedBull': 20, 'Sprite': 21, 'Starbucks': 22, 'Texaco': 23, 
    'Unicef': 24, 'Vodafone': 25, 'Yahoo': 26
}

# Reverse the dictionary to map indices to class names
index_to_class = {v: k for k, v in class_labels.items()}


def model_init():
    xcep_model = XceptionNet(num_classes=config.NUM_CLASSES,lr=config.LR)
    checkpoint = torch.load('Xception.ckpt')
    xcep_model.load_state_dict(checkpoint['state_dict'])
    
    mobile_model = MobileNetV2(num_classes=config.NUM_CLASSES,lr=config.LR)
    checkpoint = torch.load('MobileNetV2.ckpt')
    mobile_model.load_state_dict(checkpoint['state_dict'])
    
    incep_model = InceptionV3(num_classes=config.NUM_CLASSES,lr=config.LR)
    checkpoint = torch.load('InceptionV3.ckpt')
    incep_model.load_state_dict(checkpoint['state_dict'])
    
    effcent_model = EfficientNet(num_classes=config.NUM_CLASSES,lr=config.LR)
    checkpoint = torch.load('EfficientNet.ckpt')
    effcent_model.load_state_dict(checkpoint['state_dict'])
    
    return [xcep_model,incep_model,mobile_model,effcent_model]
    

models = model_init()

def inference(cnnmodel,image_path):
        
    image = Image.open(image_path).convert('RGB')
    
    cnnmodel.eval()
    input_tensor = transform_norm(image).unsqueeze(0)  # Add batch dimension
    
    # Measure inference time
    start_time = time.time()
    
    with torch.no_grad():
        output = cnnmodel(input_tensor)
        probs = torch.exp(output)
        # Get the predicted class index
        index = torch.argmax(output).item()
        
    end_time = time.time()
    inference_time = end_time - start_time
        
    predicted_class = index_to_class[index]
    predprob = round(probs[0][index].item(),2)
    if(predprob < 0.80) : predicted_class = 'None'
    
    print(f"Predicted Class: {predicted_class}")
    print(f"Accuracy: {predprob}")
    print(f"Inference Time: {inference_time:.4f} seconds")
    


for model in models:
    inference(model,image_path='nike.jpg')
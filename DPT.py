from transformers import pipeline,AutoImageProcessor, AutoModelForDepthEstimation
import torch
import numpy as np
from PIL import Image
def depth_estimation(check_point:str,image_dir:str,device:torch.device):
    depth_estimator = pipeline(task="depth-estimation", model=check_point)
    image = Image.open(image_dir)
    predictions = depth_estimator(image)
    image_processor = AutoImageProcessor.from_pretrained(check_point,force_download=True)
    model = AutoModelForDepthEstimation.from_pretrained(check_point,force_download=True)
    pixel_values = image_processor(image, return_tensors="pt").pixel_values
    pixel_values=pixel_values.to(device)
    model=model.to(device)
    with torch.no_grad():
        outputs = model(pixel_values)
        predicted_depth = outputs.predicted_depth
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    ).squeeze()
    output = prediction.cpu().numpy()
    formatted = (output * 255 / np.max(output)).astype("uint8")
    return np.array(formatted)
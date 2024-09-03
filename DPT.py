from transformers import pipeline, AutoImageProcessor, AutoModelForDepthEstimation
import torch
import numpy as np
import cv2

def depth_estimation(check_point: str, image_dir: str, device: torch.device)->np.ndarray:
 #   depth_estimator = pipeline(task="depth-estimation", model=check_point)
    image = cv2.cvtColor(src=cv2.imread(image_dir),code=cv2.COLOR_BGR2RGB)
#    predictions = depth_estimator(image)
    h, w, c = image.shape

    image_processor = AutoImageProcessor.from_pretrained(check_point, force_download=False)
    model = AutoModelForDepthEstimation.from_pretrained(check_point, force_download=False)
    pixel_values = image_processor(image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)
    model = model.to(device)
    with torch.no_grad():
        outputs = model(pixel_values)
        predicted_depth = outputs.predicted_depth
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=(h,w),
        mode="bicubic",
        align_corners=False,
    ).squeeze()
    output = prediction.cpu().numpy()
    return (output * 255 / np.max(output)).astype("uint8")

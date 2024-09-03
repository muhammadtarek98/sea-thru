import transformers
import cv2
import torch
import numpy as np


def generate_depth_map_depth_anything(image_file: str) -> np.ndarray:
    image = cv2.cvtColor(src=cv2.imread(filename=image_file), code=cv2.COLOR_BGR2RGB)
    h,w,c=image.shape
    #print(h," ",w)
    #pipeline = transformers.pipeline(task="depth-estimation",model="depth-anything/Depth-Anything-V2-Small-hf")
    #depth_map = pipeline(image)["depth"]
    image_processor = transformers.AutoImageProcessor.from_pretrained(
        pretrained_model_name_or_path="depth-anything/Depth-Anything-V2-Small-hf")
    model = transformers.AutoModelForDepthEstimation.from_pretrained(
        pretrained_model_name_or_path="depth-anything/Depth-Anything-V2-Small-hf")
    input_tensor = image_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        output = model(**input_tensor)
        depth_map = output.predicted_depth
    prediction_depth = torch.nn.functional.interpolate(
        depth_map.unsqueeze(1),
        size=(h,w),
        mode="bicubic",
        align_corners=False,
    )
    prediction_depth = prediction_depth.squeeze().cpu().numpy()
    return (prediction_depth * 255 / np.max(prediction_depth)).astype("uint8")
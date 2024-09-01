import numpy as np
import torch
import cv2
import albumentations as A
import os
from albumentations.pytorch import ToTensorV2
from DPT import depth_estimation

def depth_estimation_generator(image: np.ndarray, model_type: str, device: torch.device) -> np.ndarray:
    transform = A.Compose(
        transforms=[
            A.ToRGB(),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ToTensorV2()
        ])
    midas_transforms = torch.hub.load(repo_or_dir="intel-isl/MiDaS", model="transforms")
    transform = midas_transforms.dpt_transform
    midas_model = torch.hub.load(repo_or_dir="intel-isl/MiDaS", model=model_type)
    midas_model.to(device=device)
    midas_model.eval()
    input_image_tensor = transform(image)  #=image)["image"].to(device)
    input_image_tensor = input_image_tensor.to(device=device)  #.unsqueeze(0)
    with torch.no_grad():
        prediction = midas_model(input_image_tensor)
        prediction_upsampling = torch.nn.functional.interpolate(
            input=prediction.unsqueeze(1),
            size=image.shape[:2],
            mode="bilinear"
        )
        prediction_upsampling = prediction_upsampling.squeeze()
    output = prediction_upsampling.detach().cpu().numpy()
    return output.astype(np.uint8)


images_root_dir: str = "/home/muahmmad/projects/Image_enhancement/Enhancement_Dataset"
checkpoint:str = "vinvino02/glpn-nyu"
for file in os.listdir(path=images_root_dir):
    image_path: str = os.path.join(images_root_dir, file)
    #image = cv2.imread(filename=image_path)
    #model_type: str = "MiDaS_small"
    device = torch.device(device="cuda" if torch.cuda.is_available() else "cpu")
    depth_map = depth_estimation(check_point=checkpoint,image_dir=image_path)
    depth_map = cv2.applyColorMap(src=depth_map, colormap=cv2.COLORMAP_HOT)
    cv2.imwrite(
    filename=f"/home/muahmmad/projects/Image_enhancement/DepthMaps/{file}".format(file),
    img=depth_map)

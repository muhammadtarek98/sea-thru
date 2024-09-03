import numpy as np
import torch
import cv2
import albumentations as A
import os
from albumentations.pytorch import ToTensorV2
from DPT import depth_estimation
from MiDaS import depth_estimation_generator
from GenerateDepthMapsDepthAnything import generate_depth_map_depth_anything
images_root_dir: str = "/home/cplus/projects/m.tarek_master/Image_enhancement/Enhancement_Dataset"
checkpoint:str = "vinvino02/glpn-nyu"
for file in os.listdir(path=images_root_dir):
    image_path: str = os.path.join(images_root_dir, file)
    #image = cv2.imread(filename=image_path)
    #model_type: str = "MiDaS_small"
    device = torch.device(device="cuda" if torch.cuda.is_available() else "cpu")
    if (file.endswith(".jpg") or file.endswith(".png")and(not file.endswith(".db"))):
        if file in os.listdir("/home/cplus/projects/m.tarek_master/Image_enhancement/DepthMaps_DPT/"):
            continue
        else:
            depth_map = depth_estimation(check_point=checkpoint,image_dir=image_path,device=device)
            #depth_map=generate_depth_map_depth_anything(image_file=image_path)
            #depth_map = cv2.applyColorMap(src=depth_map, colormap=cv2.COL)
            cv2.imwrite(
            filename=f"/home/cplus/projects/m.tarek_master/Image_enhancement/DepthMaps_DPT/{file}".format(file),
            img=depth_map)
            print(f"image {file} done ")
    else:
        continue
import cv2
import torch
import numpy as np


def estimate_backscatter(image: np.ndarray, percentile: float, device: torch.device) -> torch.Tensor:
    backscatter = np.percentile(image, percentile, axis=(0, 1))
    return torch.tensor(backscatter, device=device, dtype=torch.float32)


def estimate_attenuation_light(image: torch.Tensor, backscatter: torch.Tensor) -> torch.Tensor:
    return image - backscatter


def image_correction(image: torch.Tensor, depth_map: torch.Tensor, beta_backscatter: torch.Tensor,
                     beta_depth_map: torch.Tensor) -> torch.Tensor:
    corrected_image = torch.zeros_like(image, dtype=torch.float32)

    for c in range(3):
        J_c = (image[:, :, c] - beta_backscatter[c]) * torch.exp(-beta_depth_map[c] * depth_map) + \
              beta_backscatter[c] * torch.exp(-beta_backscatter[c] * depth_map)
        corrected_image[:, :, c] = J_c

    return torch.clamp(corrected_image, min=0, max=255).to(dtype=torch.uint8)


def sea_thru_algorithm(image: np.ndarray, percentile: float, depth_map: np.ndarray, beta_backscatter: list,
                       beta_depth_map: list, device: torch.device) -> np.ndarray:
    # Convert inputs to tensors and move to the specified device
    image = torch.tensor(image, device=device, dtype=torch.float32)
    depth_map = torch.tensor(depth_map, device=device, dtype=torch.float32)

    # Estimate backscatter and attenuation light
    backscatter = estimate_backscatter(image=image.cpu().numpy(), percentile=percentile, device=device)
    attenuation_light = estimate_attenuation_light(image=image, backscatter=backscatter)

    # Convert beta parameters to tensors
    beta_backscatter = torch.tensor(beta_backscatter, device=device, dtype=torch.float32)
    beta_depth_map = torch.tensor(beta_depth_map, device=device, dtype=torch.float32)

    # Perform image correction
    corrected_image = image_correction(image=attenuation_light, depth_map=depth_map, beta_backscatter=beta_backscatter,
                                       beta_depth_map=beta_depth_map)

    return corrected_image.cpu().numpy()


if __name__ == "__main__":
    depth_map_dir_DAT = "/home/cplus/projects/m.tarek_master/Image_enhancement/DepthMaps_DepthAnything/7393_NF2_f000060.jpg"
    depth_map_dir_DPT = "/home/cplus/projects/m.tarek_master/Image_enhancement/DepthMaps_DPT/7393_NF2_f000060.jpg"
    image_dir = "/home/cplus/projects/m.tarek_master/Image_enhancement/Enhancement_Dataset/7393_NF2_f000060.jpg"

    image = cv2.imread(image_dir)
    depth_DAT = cv2.imread(depth_map_dir_DAT, cv2.IMREAD_GRAYSCALE) / 255.0
    depth_DPT = cv2.imread(depth_map_dir_DPT, cv2.IMREAD_GRAYSCALE) / 255.0

    # Parameters
    beta_depth = [0.2, 0.3, 0.4]
    beta_backscatter = [0.0001, 0.002, 0.0003]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Process images using Sea-Thru algorithm
    corrected_image_DAT = sea_thru_algorithm(image=image,
                                             depth_map=depth_DAT,
                                             percentile=10.0,
                                             device=device,
                                             beta_backscatter=beta_backscatter,
                                             beta_depth_map=beta_depth)
    corrected_image_DPT = sea_thru_algorithm(image=image,
                                             depth_map=depth_DPT,
                                             percentile=10.0,
                                             device=device,
                                             beta_backscatter=beta_backscatter,
                                             beta_depth_map=beta_depth)

    # Save the corrected images
    cv2.imwrite(filename="7393_NF2_f000060_dat.jpg",
                img=corrected_image_DAT)
    cv2.imwrite(filename="7393_NF2_f000060_dpt.jpg",
                img= corrected_image_DPT)

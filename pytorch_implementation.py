from PIL import Image
import numpy as np
import torch
from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim as optim


def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.ToTensor()
    return transform(image).unsqueeze(0)  # Add batch dimension


def load_depth_map(depth_path):
    depth = Image.open(depth_path).convert('L')  # Convert to grayscale
    transform = transforms.ToTensor()
    return transform(depth).unsqueeze(0)  # Add batch dimension


def save_image(tensor, filename):
    image = tensor.squeeze().clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
    im = Image.fromarray((image * 255).astype(np.uint8))
    im.save(filename)


class SeaThruModel(nn.Module):
    def __init__(self, depth_map, init_beta_d=0.5, init_beta_b=0.5):
        super(SeaThruModel, self).__init__()
        self.depth_map = depth_map
        self.beta_d = nn.Parameter(torch.tensor(init_beta_d),requires_grad=True)  # Attenuation coefficient
        self.beta_b = nn.Parameter(torch.tensor(init_beta_b),requires_grad=True)  # Backscatter coefficient
        self.B_inf = nn.Parameter(torch.tensor(0.1),requires_grad=True)  # Backscatter saturation value

    def forward(self, I_c):
        AL_c = I_c / (torch.exp(-self.beta_d * self.depth_map))
        BS_c = self.B_inf * (1 - torch.exp(-self.beta_b * self.depth_map))
        I_c_model = AL_c - BS_c
        return I_c_model, AL_c, BS_c


image_path = '/home/cplus/projects/m.tarek_master/Image_enhancement/depth_maps/images/000224_224_left.jpg'
depth_path = '/home/cplus/projects/m.tarek_master/Image_enhancement/depth_maps/depth_images/000224_224_left.png'

I_c = load_image(image_path)  # RGB image
I_c.requires_grad_=True
depth_map = load_depth_map(depth_path)  # Depth map
model = SeaThruModel(depth_map)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
loss_fn = nn.L1Loss()
global I_c_model
num_epochs = 10000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    I_c_model, AL_c, BS_c = model(I_c)
    loss = loss_fn(I_c_model, I_c)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item()}')
print(AL_c)
print(BS_c)
save_image(tensor=AL_c.detach(),filename="AL_c.jpg")
save_image(I_c_model.detach(), filename='I_c_model.jpg')

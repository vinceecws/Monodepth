import torch
import numpy as np
import torch.nn.functional as F

def apply_disparity(img, disp):

    N, C, H, W = img.size()

    mesh_x, mesh_y = torch.tensor(np.meshgrid(np.linspace(0, 1, W), np.linspace(0, 1, H), indexing='xy')).type_as(img)
    mesh_x = mesh_x.repeat(N, 1, 1)
    mesh_y = mesh_y.repeat(N, 1, 1)

    #grid is (N, H, W, 2)
    grid = torch.stack((mesh_x + disp.squeeze(), mesh_y), 3) 

    #grid must be in range [-1, 1]
    output = F.grid_sample(img, grid * 2 - 1, mode='bilinear', padding_mode='zeros')

    return output
from os.path import join
import random
import torch
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from PIL import Image
from torch.utils.data import Dataset

#size: (320, 1216, 3)

class KITTI(Dataset):
    def __init__(self, raw_text_dir, raw_images_dir, test=False):
        self.size = (256, 512)
        self.test = test
        self.raw_text_dir = raw_text_dir
        self.raw_images_dir = raw_images_dir
        self.image_transform = transforms.Compose([
            transforms.Resize(self.size),
            transforms.ToTensor(),
            ])
        with open(self.raw_text_dir, 'r') as text:
            self.images_dir = text.readlines()

    def __len__(self):

        return len(self.images_dir)

    def __getitem__(self, idx):
        image_pair = self.images_dir[idx].split(' ')

        image_l = Image.open(join(self.raw_images_dir, image_pair[0][:-4] + ".png"))
        image_r = Image.open(join(self.raw_images_dir, image_pair[1][:-5] + ".png"))

        if not self.test:
            image_l, image_r = image_augmentation(image_l, image_r)

        image_l = self.image_transform(image_l)
        image_r = self.image_transform(image_r)

        return(image_l, image_r)

def image_augmentation(left_img, right_img):
    random_gamma = random.uniform(0.8, 1.2)
    random_brightness = random.uniform(0.5, 2.0)
    random_flip = random.uniform(0.0, 1.0)

    TF.adjust_gamma(left_img, random_gamma)
    TF.adjust_gamma(right_img, random_gamma)

    TF.adjust_brightness(left_img, random_brightness)
    TF.adjust_brightness(right_img, random_brightness)

    if random_flip > 0.5:
        temp_img = TF.hflip(left_img)
        right_img = TF.hflip(right_img)
        left_img = right_img
        right_img = temp_img

    return left_img, right_img

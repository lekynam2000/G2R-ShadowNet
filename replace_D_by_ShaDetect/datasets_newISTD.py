import glob
import random
import os

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


def flip(image,mask):
    # Random horizontal flipping
    if random.random() > 0.5:
        image = TF.hflip(image)
        mask = TF.hflip(mask)
    # Random vertical flipping
    if random.random() > 0.5:
        image = TF.vflip(image)
        mask = TF.vflip(mask)
    return image,mask

def crop(image,mask):
    i, j, h, w = transforms.RandomCrop.get_params(
        image, output_size=(400, 400))
    image = TF.crop(image, i, j, h, w)
    mask = TF.crop(mask, i, j, h, w)
    return image, mask

def shift(tensor,shift_right,shift_up,pad=-1.0):
    tensor_shifted = torch.full_like(tensor,pad)
    _,h,w = tensor.shape
    wc = w-abs(shift_right)
    hc = h-abs(shift_up)
    x0,y0 = max(0,shift_right),max(0,shift_up)
    x1,y1 = max(0,-shift_right),max(0,-shift_up)
    tensor_shifted[:,y0:y0+hc,x0:x0+wc] = tensor[:,y1:y1+hc,x1:x1+wc]
    return tensor_shifted

class ImageDataset(Dataset):
    def __init__(self, root, mode='train'):
        self.normalizeA = transforms.Compose([
            transforms.ToTensor(),
			transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.normalizeB = transforms.Compose([
            transforms.ToTensor(),
			transforms.Normalize((0.5),(0.5))
        ])

        self.files_A = sorted(glob.glob(os.path.join(root, '%s/train_A' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%s/train_B' % mode) + '/*.*'))
        assert len(self.files_A) == len(self.files_B)

    #{A1,B1,A2,B2}
    def __getitem__(self, index):
        
        item_A1 = Image.open(self.files_A[index % len(self.files_A)])
        item_B1 = Image.open(self.files_B[index % len(self.files_B)])
        item_A1,item_B1 = crop(item_A1,item_B1)
        item_A1,item_B1 = flip(item_A1,item_B1)
        item_B1 = TF.to_grayscale(item_B1)
        item_A1 = self.normalizeA(item_A1)
        item_B1 = self.normalizeB(item_B1)

        index_2 = random.randint(0, len(self.files_A) - 1)
        item_A2 = Image.open(self.files_A[index_2])
        item_B2 = Image.open(self.files_B[index_2])
        item_A2,item_B2 = crop(item_A2,item_B2)
        item_A2,item_B2 = flip(item_A2,item_B2)
        item_B2 = TF.to_grayscale(item_B2)
        item_A2 = self.normalizeA(item_A2)
        item_B2 = self.normalizeB(item_B2)

        u12_mask,i12_mask,u21_mask,i21_mask = self.shift_and_combine(item_B1,item_B2)

        return {'A1': item_A1, 'B1': item_B1, 'A2':item_A2, 'B2':item_B2, "u12_mask":u12_mask,"i12_mask":i12_mask, "u21_mask":u21_mask, "i21_mask":i21_mask}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

    
    def shift_and_combine(self, mask1, mask2):
        # print(f"mask1.size: {mask1.size()}")
        # print(f"mask2.size: {mask2.size()}")
        assert mask1.size() == mask2.size()
        
        _,w,h = mask1.size()
        shift_right = random.randint(-int(w*0.5),int(w*0.5))
        shift_up = random.randint(-int(h*0.5),int(h*0.5))

        mask1_shifted = shift(mask1,shift_right,shift_up)
        mask2_shifted = shift(mask2,-shift_right,-shift_up)

        u12_mask = torch.max(mask1,mask2_shifted)
        i12_mask = torch.min(mask1,mask2_shifted)
        u21_mask = torch.max(mask2,mask1_shifted)
        i21_mask = torch.min(mask2,mask1_shifted)

        return u12_mask,i12_mask,u21_mask,i21_mask 


    


    
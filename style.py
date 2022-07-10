# import module
import os
import glob
import random
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch import optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import numpy as np
import logging
from tqdm import tqdm


# seed setting
def same_seeds(seed):
    # Python built-in random module
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

same_seeds(2022)

# prepare for CrypkoDataset

class CrypkoDataset(Dataset):
    def __init__(self, fnames, transform):
        self.transform = transform
        self.fnames = fnames
        self.num_samples = len(self.fnames)

    def __getitem__(self,idx):
        fname = self.fnames[idx]
        img = torchvision.io.read_image(fname)
        img = self.transform(img)
        return img

    def __len__(self):
        return self.num_samples

def get_dataset(root):
    fnames = glob.glob(os.path.join(root, '*'))
    compose = [
        transforms.ToPILImage(),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]
    transform = transforms.Compose(compose)
    dataset = CrypkoDataset(fnames, transform)
    return dataset

temp_dataset = get_dataset('/home/eegroup/eefrank/b08202017/ML_HW6/faces')

os.system('stylegan2_pytorch --data /home/eegroup/eefrank/b08202017/ML_HW6/faces --name ML_HW6_GAN --results_dir /home/eegroup/eefrank/b08202017/ML_HW6/result --models_dir /home/eegroup/eefrank/b08202017/ML_HW6/model --num-train-steps 75000')

from torchvision.utils import save_image
from stylegan2_pytorch import ModelLoader

loader = ModelLoader(
    base_dir = '/home/eegroup/eefrank/b08202017/ML_HW6',   # path to where you invoked the command line tool
    name = 'ML_HW6_GAN'                             # the project name, defaults to 'default'
)


for i in range(1000):
    noise = torch.randn(1, 512).cuda() # noise
    styles = loader.noise_to_styles(noise, trunc_psi = 0.75)
    images = loader.styles_to_images(styles)
    save_image(images, f'/home/eegroup/eefrank/b08202017/ML_HW6/output/{i+1}.jpg')
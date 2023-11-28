import glob
import random
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.utils.data as data


def populate_train_list(images_path):
	train_list = glob.glob(images_path + "*.png")
	random.shuffle(train_list)
	return train_list


class lowlight_loader(data.Dataset):
	def __init__(self, lowlight_images_path, normallight_images_path):
		self.train_list_low = populate_train_list(lowlight_images_path)
		self.train_list_normal = populate_train_list(normallight_images_path)
		self.size = 256
		print("Total training examples:", len(self.train_list_low))

	def __getitem__(self, index):
		data_lowlight_path, data_normallight_path = self.train_list_low[index], self.train_list_normal[index]
		data_lowlight, data_normallight = Image.open(data_lowlight_path), Image.open(data_normallight_path)

		data_lowlight = data_lowlight.resize((self.size,self.size), Image.ANTIALIAS)
		data_normallight = data_normallight.resize((self.size,self.size), Image.ANTIALIAS)
		data_lowlight, data_normallight = (np.asarray(data_lowlight)/255.0), (np.asarray(data_normallight)/255.0)
		data_lowlight, data_normallight = torch.from_numpy(data_lowlight).float(), torch.from_numpy(data_normallight).float()
		
		return data_lowlight.permute(2,0,1), data_normallight.permute(2,0,1)

	def __len__(self):
		return len(self.train_list_low)

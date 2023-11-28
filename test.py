import os
import glob
import numpy as np
from PIL import Image
import torch
import torchvision

import model
from option import *
from thop import profile
import time


os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Tester():
    def __init__(self):
        self.scale_factor = 1
        self.device_ids = [0, 1]
        self.net = model.GPE_Enhance(self.scale_factor).to(device)
        self.net.load_state_dict({k.replace('module.',''):v for k,v in torch.load(args.weight_dir).items()})

    def inference(self, image_path):
        # Read image from path
        data_lowlight = Image.open(image_path)
        data_lowlight = (np.asarray(data_lowlight) / 255.0)
        data_lowlight = torch.from_numpy(data_lowlight).float()
        data_lowlight = data_lowlight.permute(2,0,1).unsqueeze(0).to(device)

        # Run-time conditions
        alpha = 1.5
        data_lowlight = alpha * data_lowlight

        # Run model inference
        self.net.eval()
        with torch.no_grad():
            enhanced_image, params_maps = self.net(data_lowlight)

        # Load result directory and save image
        image_path = image_path.replace('test_data', args.output_dir)
        result_path = os.path.join(args.output_dir, image_path.split("/")[-1])
        torchvision.utils.save_image(enhanced_image, result_path)

    def test(self):
        self.net.eval()
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        
        test_list = glob.glob(args.input_dir + "*.png")
        for image_path in test_list:
            print(image_path)
            self.inference(image_path)

        print("\nTesting finished!")


if __name__ == '__main__':
    t = Tester()
    t.test()

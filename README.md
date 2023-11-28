# Semantic-aware Low-light Image Enhancement
This is the PyTorch implementation for our work "**Semantic-aware Gamma Correction for Unsupervised Low-light Image Enhancement**".<br />
[Paper Link](https://ieeexplore.ieee.org/document/10095394)

# Abstract

Low-light image enhancement aims to improve image quality and generate richer information for downstream vision tasks, receiving much research attention due to growing applications and demanding hardware requirements in smartphone photography. In this work, we highlight that **semantic understanding** is crucial to the enhancement process. While existing methods have successfully developed unsupervised or zero-shot learning techniques, they do not encourage semantic awareness and exhibit limited generalization ability for real-world images. We extend the conventional gamma correction method and propose to learn a **semantic-aware parameter map** in an **unsupervised** manner, as we do not require any paired images or segmentation labels during training. With our proposed semantics-oriented adversarial learning and semantic-aware losses, our method brings favorable enhancement results and better generalization capability over state-of-the-art approaches. Furthermore, we demonstrate that our gamma correction mapping is efficient during inference due to the lightweight parameter estimation model.

# Get Started

## 1. Requirements
* CUDA 10.0
* Python 3.8+
* PyTorch
* torchvision
* opencv-python
* numpy
* pillow
* scipy
* scikit-image
* scikit-learn
* lpips

To install all required packages (listed in the requirements.txt), run in terminal:
```
pip install -r requirements.txt
```

## 2. Prepare Datasets
### Dark CityScapes (mm20data)
Download the data from [SGZ](https://github.com/ShenZheng2000/Semantic-Guided-Low-Light-Image-Enhancement).
Create a folder `data` and unzip the dataset under the `data` folder.

# Training

**Note:** We use distributed training on **two** Tesla V100 **32GB** GPUs.

## Important Hyperparameters
Please refer to `option.py` for more details.

| Name                 | Type  | Default   |
|----------------------|-------|-----------|
| lr                   | float | 1e-4      |
| weight_decay         | float | 1e-4      |
| num_epochs           | int   | 200       |
| train_batch_size     | int   | 32        |
| num_workers          | int   | 8         |
| scale_factor         | int   | 1         |

## Run Training
Before training, make sure the dataset (mm20data) is well prepared under `data/`.

For model training, run in terminal:
```
python train.py --snapshots_folder weight_folder --pretrain_dir pretrained_weight_file
```

- `weight_folder` - directory to save models and checkpoints (default: `weight/`)
- `pretrained_weight_file` - specified if you would like to load pretrained weights for initialization (default: `weight/base.pth`)

For example, if you want to train **from scratch** and save weights under `weight/`, run in terminal:
```
python train.py
```

# Testing and Evaluation

## Run Testing
For model testing, run in terminal 
```
python test.py --weight_dir pretrained_weight_file --input_dir input_folder --output_dir output_folder
```

- `pretrained_weight_file` - model file for testing (default: `weight/base.pth`)
- `input_folder` - directory to input images (default: `data/mm20data/test/test_L/`)
- `output_folder` - directory to save output images (default: `data/test_output/`)

For example, if your want to use our trained model `weight/base.pth` for testing low-light images under `data/mm20data/test/test_L/`, and save output images under `data/test_output/`, you can run in terminal:
```
python test.py --weight_dir weight/base.pth --input_dir data/mm20data/test/test_L/ --output_dir data/test_output/
```

## Run Evaluation
For quantitative image quality evaluation after testing, run in terminal:
```
python evaluate.py --output_path path_to_outputs --ref_path path_to_gt
```

- `path_to_outputs` - directory of output images (eg., `data/test_output/`)
- `path_to_gt` - directory of ground-truth images (eg., `data/mm20data/test/test_H/`)

**Note:** Ground-truth images should have the same filename prefix as the corresponding output images.
**Note:** We use model parameters stored in `niqe/mvg_params.mat` for NIQE evaluation.

For example, if your want to evaluate output images under `data/test_output/` with ground-truth images under `data/mm20data/test/test_H/`, you can run in terminal:
```
python evaluate.py --output_path data/test_output/ --ref_path data/mm20data/test/test_H/
```

## Quantitative Metric Results
We list the metric results for input images under `data/mm20data/test/test_L/` with our trained model `weight/base.pth`.

| PSNR  | SSIM  | LPIPS | NIQE  |
|-------|-------|-------|-------|
| 23.25 | 0.852 | 0.204 | 4.988 |

# Reference
This codebase is heavily based upon [SGZ](https://github.com/ShenZheng2000/Semantic-Guided-Low-Light-Image-Enhancement).

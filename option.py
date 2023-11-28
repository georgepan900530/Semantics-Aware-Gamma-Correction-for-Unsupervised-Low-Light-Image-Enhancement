import argparse

parser = argparse.ArgumentParser(description="Train_Test")

### Training Settings
parser.add_argument('--lowlight_images_path', type=str, default="data/mm20data/train/train_L/")
parser.add_argument('--normallight_images_path', type=str, default="data/mm20data/train/train_H/")
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--weight_decay', type=float, default=0.0001)
parser.add_argument('--num_epochs', type=int, default=200)
parser.add_argument('--train_batch_size', type=int, default=32)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--scale_factor', type=int, default=1)
parser.add_argument('--display_iter', type=int, default=10)
parser.add_argument('--snapshot_iter', type=int, default=10)
parser.add_argument('--snapshots_folder', type=str, default="weight/")
parser.add_argument('--load_pretrain', type=bool, default=False)
parser.add_argument('--pretrain_dir', type=str, default="weight/base.pth")

### Segmentation Model Settings
parser.add_argument("--num_of_SegClass", type=int, default=19)
parser.add_argument("--seg_ckpt", type=str, default='seg_ckpt/best_deeplabv3plus_mobilenet_cityscapes_os16.pth')

### Testing Settings
parser.add_argument('--weight_dir', type=str, default="weight/base.pth")
parser.add_argument('--input_dir', type=str, default='data/mm20data/test/test_L/')
parser.add_argument('--output_dir', type=str, default='data/test_output/')

args = parser.parse_args()

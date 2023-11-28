import os
import torch.nn as nn
import torch.optim
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
from torchvision import transforms

import model
import seg_network
import loss
import dataloader
from option import *


gpu_list = [0,1]
gpu_list_str = ','.join(map(str, gpu_list))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu_list_str) # GPU only
device = torch.device("cuda:0")


class Trainer():
    def __init__(self):
        self.scale_factor = args.scale_factor
        self.net = model.GPE_Enhance(self.scale_factor)
        self.net = nn.DataParallel(self.net, device_ids = gpu_list)
        self.net.to(device)

        self.Discriminator = model.Discriminator(img_size=(256, 256, args.num_of_SegClass), dim=16)
        self.Discriminator = nn.DataParallel(self.Discriminator, device_ids = gpu_list)
        self.Discriminator.to(device)

        self.train_dataset = dataloader.lowlight_loader(args.lowlight_images_path, args.normallight_images_path)
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                                       batch_size=args.train_batch_size,
                                                       shuffle=True,
                                                       num_workers=args.num_workers,
                                                       pin_memory=True)

        self.losses = {'G': [], 'D': [], 'GP': [], 'gradient_norm': []}
        self.L_color = loss.L_color()
        self.L_brightness = loss.L_brightness()
        self.L_smooth = loss.L_smooth()
        self.L_flip = nn.MSELoss(reduction='sum')
        self.L_recon = nn.MSELoss(reduction='mean')

        self.G_optimizer = torch.optim.Adam(self.net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.D_optimizer = torch.optim.Adam(self.Discriminator.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        self.num_steps = 0
        self.gp_weight = 10
        self.critic_iterations = 10
        self.num_epochs = args.num_epochs
        self.display_iter = args.display_iter
        self.snapshot_iter = args.snapshot_iter
        self.snapshots_folder = args.snapshots_folder

        # pretrained segmentation model
        self.seg_model = seg_network.modeling.__dict__['deeplabv3plus_mobilenet'](num_classes=args.num_of_SegClass, output_stride=16)
        for m in self.seg_model.backbone.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.momentum = 0.01
        self.seg_model.load_state_dict(torch.load(args.seg_ckpt, map_location=torch.device('cpu'))["model_state"])
        self.seg_model = nn.DataParallel(self.seg_model, device_ids=gpu_list)
        self.seg_model.to(device)
        print("Resume seg model from %s" % args.seg_ckpt)

        if args.load_pretrain == True:
            self.net.load_state_dict(torch.load(args.pretrain_dir, map_location=device))
            print(f"Pretrained weight is OK, Loading weight from {args.pretrain_dir}")


    def get_loss(self, A, enhanced_image, seg_map):
        loss_smooth = self.L_smooth(A, seg_map)
        class_brightness = self.L_brightness(enhanced_image, seg_map)
        class_color = self.L_color(enhanced_image, seg_map)

        return loss_smooth, class_brightness, class_color
    

    def _gradient_penalty(self, real_data, generated_data):
        batch_size = real_data.size()[0]

        # Calculate interpolation
        alpha = torch.rand(batch_size, 1, 1, 1)
        alpha = alpha.expand_as(real_data).to(device)
        interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
        interpolated = Variable(interpolated, requires_grad=True).to(device)

        # Calculate probability of interpolated examples
        prob_interpolated = self.Discriminator(interpolated)

        # Calculate gradients of probabilities with respect to examples
        gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(prob_interpolated.size()).to(device),
                               create_graph=True, retain_graph=True)[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(batch_size, -1)
        self.losses['gradient_norm'].append(gradients.norm(2, dim=1).mean().data)

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # Return gradient penalty
        return self.gp_weight * ((gradients_norm - 1) ** 2).mean()
    

    def train(self):
        self.net.train()
        self.seg_model.eval()
        self.Discriminator.train()
        print(self.net)
        print(self.Discriminator)

        for epoch in range(self.num_epochs):
            print(f"\n[Epoch {epoch+1} / {self.num_epochs}]\n")
            for iteration, (img_lowlight, img_normal) in enumerate(self.train_loader):
                self.num_steps += 1

                img_lowlight = Variable(img_lowlight).to(device)
                img_normal = Variable(img_normal).to(device)

                with torch.set_grad_enabled(True):
                    # obtain enhanced image
                    enhanced_image, A = self.net(img_lowlight)
                    enhanced_normal, _ = self.net(img_normal)

                    # self-supervised flip consistency
                    flip_enhanced_image, flip_A = self.net(transforms.RandomHorizontalFlip(p=1)(img_lowlight))
                    loss_flip = self.L_flip(A, transforms.RandomHorizontalFlip(p=1)(flip_A))

                    # reference image reconstruction
                    loss_recon = self.L_recon(img_normal, enhanced_normal)

                    # get seg map of enhanced image and low light image
                    fake_seg = self.seg_model(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(enhanced_image))
                    real_seg = self.seg_model(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img_normal))

                    # Calculate probabilities on real and generated seg output
                    d_generated = self.Discriminator(fake_seg)
                    d_real = self.Discriminator(real_seg)

                    # Get gradient penalty
                    gradient_penalty = self._gradient_penalty(real_seg, fake_seg)
                    self.losses['GP'].append(gradient_penalty.data)

                    # Create loss and update D
                    d_loss = d_generated.mean() - d_real.mean() + gradient_penalty
                    self.D_optimizer.zero_grad()
                    d_loss.backward(retain_graph=True)
                    self.D_optimizer.step()

                    # Calculate loss and update G
                    loss_smooth, brightness_e, color_e = self.get_loss(A, enhanced_image, fake_seg)
                    _, brightness_n, color_n  = self.get_loss(A, img_normal, real_seg)
                    loss_brightness = torch.mean(torch.pow(brightness_e - brightness_n, 2))
                    loss_color = torch.mean(torch.pow(color_e - color_n, 2))
                    g_loss = 20 * loss_brightness + 10 * loss_color + 3 * loss_smooth + 0.5 * loss_flip + loss_recon
                    
                    # Only record WGAN loss for g every |critic_iterations| iterations
                    if self.num_steps % self.critic_iterations == 0:
                      g_loss -= d_generated.detach().clone().mean()

                    self.G_optimizer.zero_grad()
                    g_loss.backward()
                    self.G_optimizer.step()

                    # Record loss
                    self.losses['D'].append(d_loss.data)
                    self.losses['G'].append(g_loss.data)

                if (iteration + 1) % self.display_iter == 0 and self.num_steps > self.critic_iterations:
                    print(f"[Iter {iteration+1}]  D: {self.losses['D'][-1]:.4f}  G: {self.losses['G'][-1]:.4f}")
                if (iteration + 1) % self.snapshot_iter == 0:
                    torch.save(self.net.state_dict(), self.snapshots_folder + "Epoch" + str(epoch+1) + '.pth')


if __name__ == "__main__":
    t = Trainer()
    t.train()

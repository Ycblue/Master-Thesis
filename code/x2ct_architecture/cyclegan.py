"""
To run this template just do:
python gan.py
After a few epochs, launch tensorboard to see the images being generated at every batch.
tensorboard --logdir default
"""
import os
from argparse import ArgumentParser
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# from torchvision.datasets import MNIST
from concat_dataset import Concat_Dataset 
from models import Generator
from models import Discriminator
from models import Vgg16
from ct2x_projection import project_series

import pytorch_lightning as pl

class Xray_Style_Generator(nn.Module):
    def __init__(self, batch_size):
        super(Generator, self).__init__()
        self.vgg16 = Vgg16()

    def forward(self, x):
        output = self.vgg16(x)

        return output
    
# class Discriminator(nn.Module):
#     def __init__(self, batch_size):
#         super(Discriminator, self).__init__()

#         self.patch3d = nn.Sequential(OrderedDict([]))
#         for i in range(3):
#             if i == 0:
#                 block1 = nn.Conv3d(batch_size,64, 4, stride=2)
#             else: block1 = nn.Conv3d(64,64, 4, stride=2)
#             block2 = nn.InstanceNorm3d(64)
#             block3 = nn.ReLU()
#             self.patch3d.add_module('conv3d%d' % (i+1), block1)
#             self.patch3d.add_module('in%d' % (i+1), block2)
#             self.patch3d.add_module('relu%d' % (i+1), block3)
#         self.patch3d.add_module('conv3d4', nn.Conv3d(64, 32, 4, stride=1))
#         self.patch3d.add_module('in4', nn.InstanceNorm3d(32))
#         self.patch3d.add_module('relu4', nn.ReLU())
#         self.patch3d.add_module('conv3d5', nn.Conv3d(32, 1, 4))
#         self.sig = nn.Sigmoid()

#     def forward(self, x):
#         # returns [1,1,8,8,8] array
#         out = self.patch3d(x)
#         out = self.sig(out)
#         return out

class GAN(pl.LightningModule):

    def __init__(self, hparams):
        super(GAN, self).__init__()
        self.hparams = hparams
        batch_size = self.hparams.batch_size
        # networks
        # mnist_shape = (1, 28, 28)
        self.generator_x2ct = Generator(batch_size)
        self.generator_ct2x = Vgg16()
        self.discriminator_ct = CT_Discriminator(batch_size)
        self.discriminator_x = X_Discriminator(batch_size)

        # cache for generated images
        self.generated_imgs = None
        self.last_imgs = None

    def forward_x2ct(self, z1, z2):
        return self.generator(z1, z2)

    def forward_ct2x(self, z):
        # do projection thing 
        # need to change package to use arrays since they're already loaded in model
        return self.generator_ct2x(z)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def g_adversarial_loss(self, d_fake):
        loss = 0.5 * torch.mean((d_fake-1)**2)
        return loss
    
    def d_adversarial_loss(self, d_real, d_fake):
        loss = 0.5 * (torch.mean((d_real-1)**2) + torch.mean((d_fake-0)**2))
        return loss

    # def reconstruction_loss(self, ):
    #     return F.mse_loss()
    def cycle_loss(self, x, x_hat):
        # i.e. loss between x and cycleGAN(x)
        loss = nn.L1Loss(x, x_hat)
        return loss
        
    def identity_loss(self, x, x_hat):
        loss = nn.L1Loss(x, x_hat)
        return loss
    
    def KLD_loss()


    def training_step(self, batch, batch_idx, optimizer_idx):
        # print()
        # if len(batch) == 3:
        #     label, img1, img2 = batch
        #     imgs = (img1, img2)
        # else:
        #     label, imgs = batch
        # print('len(batch[imgs])\n', len(batch['imgs']))
        # label = batch['label']
        imgs = batch

        self.last_imgs = imgs
        # print(imgs[0])
        # print(imgs[1])
        # print('Label: \n', label)
        # train generator
        (z1, z2) = torch.split(imgs['X'], 1, 1)
        ct = imgs['CT']
        if optimizer_idx == 0:
            # sample noise
            # change to image input
            # print('imgs.shape: \n', imgs.shape)
            
            
            # print('z1: \n', z1)
            # print('z2: \n', z2)
            # z1 = torch.randn(imgs.shape[0], self.hparams.latent_dim)
            # z2 = torch.randn(imgs.shape[0], self.hparams.latent_dim)
            # match gpu device (or keep as cpu)
            # if self.on_gpu:
            #     z1 = z1.cuda(imgs.device.index)
            #     z2 = z2.cuda(imgs.device.index)

            # generate images
            self.generated_imgs = self.forward(z1, z2)

            # log sampled images
            # sample_imgs = self.generated_imgs[:6]
            # grid = torchvision.utils.make_grid(sample_imgs)
            # self.logger.experiment.add_image('generated_images', grid, 0)

            # ground truth result (ie: all fake)
            # put on GPU because we created this tensor inside training_loop
            # valid = torch.ones(imgs.size(0), 1)
            # valid = torch.ones([1,1,8, 8, 8], 1)
            # if self.on_gpu:
            #     valid = valid.cuda(imgs.device.index)

            # adversarial loss is binary cross-entropy
            # g_loss = self.adversarial_loss(self.discriminator(self.generated_imgs), valid)
            g_loss = self.g_adversarial_loss(self.discriminator(self.generated_imgs))
            # g_loss = self.adversarial_loss(self.discriminator(self.generated_imgs), valid)
            tqdm_dict = {'g_loss': g_loss}
            output = OrderedDict({
                'loss': g_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            # print(output)
            # z1.cpu()
            # z2.cpu()
            # imgs.cpu()
            return output

        # train discriminator
        if optimizer_idx == 1:
            # Measure discriminator's ability to classify real from generated samples
            
            # how well can it label as real?
            # valid = torch.ones([1,1,8,8,8], 1)
            # if self.on_gpu:
                # valid = valid.cuda(imgs.device.index)

            # real_loss = self.adversarial_loss(self.discriminator(imgs), valid)
            d_loss = self.d_adversarial_loss(self.discriminator(ct), self.generated_imgs.detach())

            # how well can it label as fake?
            # fake = torch.zeros([1,1,8,8,8], 1)
            # if self.on_gpu:
            #     fake = fake.cuda(imgs.device.index)

            # fake_loss = self.adversarial_loss(
            #     self.discriminator(self.generated_imgs.detach()), fake)

            # discriminator loss is the average of these
            # d_loss = (real_loss + fake_loss) / 2
            tqdm_dict = {'d_loss': d_loss}
            output = OrderedDict({
                'loss': d_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)
        return {'val_loss': F.cross_entropy(y_hat, y)(y_hat, y)}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'avg_val_loss': avg_loss}
        
    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []

    @pl.data_loader
    def train_dataloader(self):
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize([0.5], [0.5])]
                                        )
        xray_root = '/work/scratch/lan/datasets/xrays/train/test'
        ct_root = '/work/scratch/lan/datasets/LIDC/train/test'
        dataset = Concat_Dataset(xray_root, ct_root)
        # dataset = MNIST(os.getcwd(), train=True, download=True, transform=transform)
        return DataLoader(dataset, batch_size=self.hparams.batch_size)

    # def on_save_checkpoint(self, checkpoint)


    # def on_epoch_end(self):
    #     z = torch.randn(8, self.hparams.latent_dim)
    #     # match gpu device (or keep as cpu)
    #     if self.on_gpu:
    #         z = z.cuda(self.last_imgs.device.index)

        # log sampled images
        # sample_imgs = self.forward(z)
        # grid = torchvision.utils.make_grid(sample_imgs)
        # self.logger.experiment.add_image(f'generated_images', grid, self.current_epoch)

    

def main(hparams):
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    model = GAN(hparams)

    # ------------------------
    # 2 INIT TRAINER
    # ------------------------
    trainer = pl.Trainer(gpus=1, default_save_path='work/scratch/lan/output/x2ct/lightning/')

    # trainer = pl.Trainer(gpus=1, default_save_path='work/scratch/lan/output/x2ct/lightning/', fast_dev_run=True)
    # track gradient norms: track_grad_norm=1
    # print tensors with nan gradients: print_nan_grads=True
    #
    # ------------------------
    # 3 START TRAINING
    # ------------------------
    trainer.fit(model)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--latent_dim", type=int, default=100,
                        help="dimensionality of the latent space")
    parser.add_argument("--gpus", type=int, default=1, help="how many gpus")
    hparams = parser.parse_args()
    # print(hparams)
    main(hparams)
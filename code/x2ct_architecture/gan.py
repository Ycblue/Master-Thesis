import os
from argparse import ArgumentParser
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as utils
from torch.utils.data import DataLoader
# from apex import amp
# from torch.utils.tensorboard import SummaryWriter

# from torchvision.datasets import MNIST
# from dataset import Dataset 
from dataset_hdf5 import Dataset_Hdf5 
from models import Generator
from models import CT_Discriminator

import pytorch_lightning as pl
from pytorch_lightning.logging.tensorboard import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import SimpleITK as sitk

class GAN(pl.LightningModule):

    def __init__(self, hparams):
        super(GAN, self).__init__()
        self.hparams = hparams
        batch_size = self.hparams.batch_size
        self.lambda_rec = self.hparams.lambda_rec
        self.lambda_adv = self.hparams.lambda_adv
        self.lambda_pl = self.hparams.lambda_pl
        # networks
        # mnist_shape = (1, 28, 28)
        self.generator = Generator(batch_size)
        self.discriminator = CT_Discriminator(batch_size)

        # cache for generated images
        self.generated_imgs = None
        self.last_imgs = None
        self.epoch = 0

    def forward(self, z1, z2):
        return self.generator(z1, z2)


    def g_adversarial_loss(self, d_fake):
        loss = 0.5 * torch.mean((d_fake-1)**2)
        # print('g_adversarial_loss: \n')
        # print(loss.type())
        # print(loss)
        # print(loss.type(torch.float32))
        # print(loss.type(torch.float16))
        return loss
    
    def d_adversarial_loss(self, d_real, d_fake):
        loss = 0.5 * (torch.mean((d_real-1)**2) + torch.mean((d_fake-0)**2))
        return loss

    def reconstruction_loss(self, y_hat, y):
        return F.mse_loss(y_hat, y)

    def projection_loss(self, y_hat, y):
        def projection(m, axis):
            assert axis > 1
            proj = torch.sum(m,axis)
            return proj

        loss = 1/3*(
            F.l1_loss(projection(y_hat,2), projection(y,2))+
            F.l1_loss(projection(y_hat,3), projection(y,3))+
            F.l1_loss(projection(y_hat,4), projection(y,4)))
        return loss

    def training_step(self, batch, batch_idx, optimizer_idx):
        x, ct = batch
        # imgs = batch

        # self.last_imgs = imgs
        # for item in imgs: 
        # (z1, z2) = torch.split(imgs['X'], 1, 1)
        (z1, z2) = torch.split(x, 1, 1)
        # ct = imgs['CT']

        self.generated_imgs = self(z1, z2)
        
        if optimizer_idx == 0:

            # generate images
            # self.generated_imgs = self(z1, z2)

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
            # loss_g_adv = self.adversarial_loss(self.discriminator(self.generated_imgs), valid)
            loss_g_adv = self.g_adversarial_loss(self.discriminator(self.generated_imgs))

            loss_rec = self.reconstruction_loss(self.generated_imgs, ct)

            loss_pl = self.projection_loss(self.generated_imgs, ct)

            g_loss = self.lambda_adv * loss_g_adv + self.lambda_rec * loss_rec + self.lambda_pl * loss_pl
            # loss_g_adv = self.adversarial_loss(self.discriminator(self.generated_imgs), valid)
            self.logger.experiment.add_scalar('Train/loss_g_adv', loss_g_adv, self.epoch)
            self.logger.experiment.add_scalar('Train/loss_rec', loss_rec, self.epoch)
            self.logger.experiment.add_scalar('Train/loss_pl', loss_pl, self.epoch)
            self.logger.experiment.add_scalar('Train/G_Loss', g_loss, self.epoch)
            tqdm_dict = {'loss_g_adv': loss_g_adv, 
                         'loss_rec': loss_rec,
                         'loss_pl': loss_pl}
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
            d_loss = self.lambda_adv * self.d_adversarial_loss(self.discriminator(ct), self.discriminator(self.generated_imgs))

            self.logger.experiment.add_scalar('Train/D_Loss', d_loss, self.epoch)
            tqdm_dict = {'d_loss': d_loss}
            output = OrderedDict({
                'loss': d_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

    def validation_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop
        :param batch:
        :return:

        If `val_dataset` is not tagged with `@pl.data_loader`, `validation_step` requires additional `optimizer_idx` argument
        """
        (x, ct) = batch
        (z1, z2) = torch.split(x, 1, 1)
        y_hat = self.discriminator(self(z1, z2))
        y = torch.squeeze(self.discriminator(ct), 0)

        loss_val = self.d_adversarial_loss(y, y_hat)

        # loss_g_adv = self.g_adversarial_loss(self.discriminator(self.generated_imgs))

        # loss_rec = self.reconstruction_loss(self.generated_imgs, ct)

        # loss_pl = self.projection_loss(self.generated_imgs, ct)

        # loss_val_g = self.lambda_adv * loss_g_adv + self.lambda_rec * loss_rec + self.lambda_pl * loss_pl


        output = OrderedDict({
            'val_loss': loss_val,
            # 'val_acc': val_acc,
        })

        # can also return just a scalar instead of a dict (return loss_val)
        return output


    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.logger.experiment.add_scalar('Val/Val_Loss', avg_loss, self.epoch)
        tqdm_dict = {'val_loss': avg_loss}
        result = {'progress_bar': tqdm_dict, 'log': tqdm_dict}
        return result
    
    def test_step(self, batch, batch_idx):
        (x, ct) = batch
        (z1, z2) = torch.split(x, 1, 1)
        fake_ct = self(z1,z2)
        
        y_hat = self.discriminator(fake_ct)
        y = torch.squeeze(self.discriminator(ct), 0)
        loss_test = self.d_adversarial_loss(y, y_hat)

        # log and test images.
        fake_ct = fake_ct.squeeze().cpu()
        grid = utils.make_grid(fake_ct[64, :, :])
        self.logger.experiment.add_image('Test_Fake_CT/{}'.format(batch_idx), grid, global_step=0)

        fake_ct = np.squeeze(self.last_imgs.numpy())
        img = sitk.GetImageFromArray(fake_ct.astype(np.uint16))
        sitk.WriteImage(img, '%s/test/%d.dcm' % (output_directory, batch_idx))

        output = OrderedDict({
            'test_loss': loss_test,
            # 'val_acc': val_acc,
        })
        return output
    
    def test_end(self, outputs):
        avg_test_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        self.logger.experiment.add_scalar('Test/Test_Loss', avg_test_loss, self.epoch)
        
        

        tqdm_dict = {'test_loss': avg_test_loss}
        result = {'progress_bar': tqdm_dict, 'log': tqdm_dict}
        return {'test_loss': avg_test_loss}

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        sched_g = torch.optim.lr_scheduler.MultiStepLR(opt_g, [50, 60, 70, 80, 90, 100], gamma=0.1, last_epoch=-1)
        sched_d = torch.optim.lr_scheduler.MultiStepLR(opt_d, [50, 60, 70, 80, 90, 100], gamma=0.1, last_epoch=-1)
        return [opt_g, opt_d], [sched_g, sched_d]
    
    def configure_apex(self, amp, model, optimizers, amp_level):
        model, optimizers= amp.initialize(
            model, optimizers, opt_level=amp_level
        )
        return model, optimizers

    @pl.data_loader
    def train_dataloader(self):
        # xray_root = self.hparams.x_root
        # ct_root = self.hparams.ct_root
        data_root = self.hparams.data_root

        
        if self.hparams.debug_mode:
            dataset = Dataset_Hdf5(data_root, mode='debug')
        else: 
            dataset = Dataset_Hdf5(data_root, mode='train')

        if self.hparams.gpus > 1:
            dist_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
            dataloader = DataLoader(dataset, batch_size=self.hparams.batch_size, sampler = dist_sampler)
        else: dataloader = DataLoader(dataset, batch_size=self.hparams.batch_size)
        # dataloader = DataLoader(dataset, batch_size=self.hparams.batch_size)

        return dataloader
    @pl.data_loader
    def val_dataloader(self):
       
        # xray_root = self.hparams.x_root
        # ct_root = self.hparams.ct_root
        data_root = self.hparams.data_root
        if self.hparams.debug_mode:
            dataset = Dataset_Hdf5(data_root, mode='debug')
        else: 
            dataset = Dataset_Hdf5(data_root, mode='val')

        if self.hparams.gpus > 1:
            dist_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
            dataloader = DataLoader(dataset, batch_size=self.hparams.batch_size, sampler = dist_sampler)
        else: dataloader = DataLoader(dataset, batch_size=self.hparams.batch_size)
        # dataloader = DataLoader(dataset, batch_size=self.hparams.batch_size)

        return dataloader

    @pl.data_loader
    def test_dataloader(self):
        data_root = self.hparams.data_root
        if self.hparams.debug_mode:
            dataset = Dataset_Hdf5(data_root, mode='debug')
        else: 
            dataset = Dataset_Hdf5(data_root, mode='test')
        if self.hparams.gpus > 1:
            dist_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
            dataloader = DataLoader(dataset, batch_size=self.hparams.batch_size, sampler = dist_sampler)
        else: dataloader = DataLoader(dataset, batch_size=self.hparams.batch_size)
        # dataloader = DataLoader(dataset, batch_size=self.hparams.batch_size)
        return dataloader


    def on_epoch_end(self):

        output_directory = self.hparams.output_path
        fake_ct = self.generated_imgs.detach().squeeze().cpu()
        grid = utils.make_grid(fake_ct[64, :, :])
        self.logger.experiment.add_image('Fake_CT/Epoch_{}'.format(self.epoch), grid, global_step=0)

        fake_ct = np.squeeze(fake_ct.numpy())
        img = sitk.GetImageFromArray(fake_ct.astype(np.uint16))
        sitk.WriteImage(img, '%s/%d.dcm' % (output_directory, self.epoch))
        
        self.epoch += 1
        # self.logger.experiment.add_image('{epoch:02d}', self.generated_imgs, 0)
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
    logger = TensorBoardLogger('{}/tb_logs'.format(hparams.log_path), name='x2ct_gan')
    version_path = '{}/{}/{}'.format(
        hparams.log_path,
        logger.name,
        hparams.file_name
    )
    checkpoint_path = '{}/{}'.format(version_path, 'checkpoints')
    checkpoint_callback = ModelCheckpoint(filepath=checkpoint_path, verbose=1) # needs work

    trainer = pl.Trainer(fast_dev_run=hparams.fast_dev_run,
                         gpus=hparams.gpus, 
                        #  gpus=[0,1],
                         distributed_backend=hparams.distributed_backend,
                         amp_level='O1',
                         use_amp=hparams.use_amp,
                         max_epochs=hparams.max_epochs,
                         min_epochs=hparams.min_epochs,
                         default_save_path=hparams.log_path,  checkpoint_callback=checkpoint_callback, 
                         logger=logger)
    # track gradient norms: track_grad_norm=1
    # print tensors with nan gradients: print_nan_grads=True
    #
    # ------------------------
    # 3 START TRAINING
    # ------------------------
    trainer.fit(model)

def get_args():
    # if any arg is set to `None`, Tensorboard will throw ValueError.
    parser = ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=32, help='size of the batches')
    parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
    parser.add_argument('--b1', type=float, default=0.5,
                        help='adam: decay of first order momentum of gradient')
    parser.add_argument('--b2', type=float, default=0.999,
                        help='adam: decay of first order momentum of gradient')
    parser.add_argument('--lambda_rec', type=float, default=10,
                        help='weight of reconstrcution loss')
    parser.add_argument('--lambda_adv', type=float, default=0.1,                  help='weight of adversarial loss')
    parser.add_argument('--lambda_pl', type=float, default=10,         help='weight of projection loss')
    parser.add_argument('--latent_dim', type=int, default=100,
                        help='dimensionality of the latent space')
    parser.add_argument('--max_epochs', type=int, default=1000, help='max epochs to be run')
    parser.add_argument('--min_epochs', type=int, default=1, help='min epochs to be run')

    # parser.add_argument('--x_root', type=str, default='/work/scratch/lan/datasets/xrays', help='xray dataset root')
    # parser.add_argument('--ct_root', type=str, default='/work/scratch/lan/datasets/LIDC', help='ct dataset root')
    parser.add_argument('--data_root', type=str, default='/work/scratch/lan/datasets/hdf5/x2ct_dataset.hdf5', help='ct dataset root')
    parser.add_argument('--log_path', type=str, default='/work/scratch/lan/output/x2ct/lightning', help='location for logs')
    parser.add_argument('--output_path', type=str, default='/work/scratch/lan/output/x2ct/fake_ct', help='location for output cts')
    parser.add_argument('--file_name', type=str, default='', help='use job name as file name')

    parser.add_argument('--gpus', type=int, default=1, help='how many gpus') 
    parser.add_argument('--distributed_backend', type=str, default='dp', help='supports three options dp, ddp, ddp2') 
    
    parser.add_argument('--fast_dev_run', type=bool, default=False,                            help='enable fast_dev_run')
    parser.add_argument('--debug_mode', type=bool, default=False, 
    help='enable debug mode')
    parser.add_argument('--use_amp', type=bool, default=False, help='use only with capable graphics card!')
    hparams = parser.parse_args()

    return hparams

if __name__ == '__main__':
    
    # print(hparams)
    main(get_args())
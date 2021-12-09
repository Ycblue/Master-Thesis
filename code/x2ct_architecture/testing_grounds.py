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
from torchvision.datasets import MNIST
import SimpleITK as sitk
from skimage.transform import resize

from models import ConnectionB

import pytorch_lightning as pl
from dataset import Dataset
from argparse import Namespace
from gan import GAN

hparams = {
    "batch_size": 1,
    "lambda_rec": 10,
    "lambda_adv": 0.1,
    "lambda_pl": 10
}
namespace = Namespace(**hparams)

ckpt_path = '/work/scratch/lan/output/x2ct/lightning/x2ct_gan/version_166/checkpoints/_ckpt_epoch_14.ckpt'

# class MyLightningModel(pl.LightningModule):
#     def __init__(self, hparams):
#         self.hparams = hparams
    
pretrained_model = GAN.load_from_checkpoint(ckpt_path)


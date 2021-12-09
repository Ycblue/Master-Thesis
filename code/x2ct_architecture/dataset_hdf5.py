from torch.utils.data import Dataset
import os
import sys
import torch
import h5py
import numpy as np
import SimpleITK as sitk
from skimage.transform import resize
from tqdm import tqdm
import torchvision.transforms as transforms
import torch.nn.functional as F

class Dataset_Hdf5(Dataset):
    """Xray and Ct dataset as dictionary from `.hdf5` file.

    Returns paired Xray arrays and one CT array as tuple.
    
    With `cache=True` all data is written to CPU RAM first, instead of loading each time a batch is called.

    Entire dataset takes about 16 GB RAM. 

    Example: 

    ```
    dataset = Dataset(xray_root, ct_root, mode='debug', cache=True)
    data = DataLoader(dataset, batch_size = 1)
    for item in data:
        (x, ct) = item
    ```

    """
    
    def __init__(self, data_root, mode, cache=True):
        
        # f = h5py.File('{}/x2ct_dataset.hdf5'.format(data_root), 'r')
        f = h5py.File(data_root, 'r')

        self.cache = cache
        # self.transform = transforms.Compose(_transforms)
        self.transform = transforms.Normalize([0.5], [0.5])

        self.samples = [] # holds cached images
        self.xray_samples = [] #list of xray arrays 
        self.ct_samples = [] #list of ct arrays 
        
        self.xray_path_samples = [] # holds path to images when cache=False
        self.ct_path_samples = [] # holds path to images when cache=False

        for key in tqdm(f['xrays/{}'.format(mode)].keys()):
            xray_path = 'xrays/{}/{}'.format(mode, key)
            self.xray_path_samples.append(xray_path)
            if self.cache:
                #resize
                xray_a = resize(f[xray_path]['1'], (1, 128, 128))
                xray_b = resize(f[xray_path]['2'], (1, 128, 128))
                #scale between 0 and 1
                xray_a = (xray_a - np.min(xray_a))/ (np.max(xray_a)-np.min(xray_a))
                xray_b = (xray_b - np.min(xray_b))/ (np.max(xray_b)-np.min(xray_b))
                # xray_a = self.transform(torch.from_numpy(xray_a))
                # xray_b = self.transform(torch.from_numpy(xray_b))
                # normalize between -1 and 1
                xray_a = transforms.functional.normalize(torch.from_numpy(xray_a), [0.5], [0.5])
                xray_b = transforms.functional.normalize(torch.from_numpy(xray_b), [0.5], [0.5])
                self.xray_samples.append(torch.cat([xray_a, xray_b], 0))

        for key in tqdm(f['LIDC/{}'.format(mode)].keys()):
            ct_path = 'LIDC/{}/{}'.format(mode, key)
            self.ct_path_samples.append(ct_path)
            if self.cache:
                ct = resize(f[ct_path], (1, 128, 128, 128))
                ct = (ct - np.min(ct))/ (np.max(ct)-np.min(ct))
                ct_array = F.normalize(torch.from_numpy(ct), p=2, dim=2)
                self.ct_samples.append(ct_array)

        if self.cache: 
            for x, ct in zip(self.xray_samples, self.ct_samples):
                self.samples.append((x, ct))

    def __len__(self):
        if self.cache:
            return len(self.samples)
        else:
            return len(self.xray_path_samples) # presuming xray and ct dataset are of equal length

    def __getitem__(self, idx):
        '''
        Returns sets of images for idx in path_samples or samples, depending on value of cache.

        '''
        if not self.cache:
            f = h5py.File('{}/x2ct_dataset.hdf5'.format(data_root), 'r')

            xray_a = resize(f[self.xray_path_samples[idx]]['1'], (1, 128, 128))
            xray_b = resize(f[self.xray_path_samples[idx]]['2'], (1, 128, 128))
            xxray_a = (xray_a - np.min(xray_a))/ (np.max(xray_a)-np.min(xray_a))
            xray_b = (xray_b - np.min(xray_b))/ (np.max(xray_b)-np.min(xray_b))
            xray_a = transforms.functional.normalize(torch.from_numpy(xray_a), [0.5], [0.5])
            xray_b = transforms.functional.normalize(torch.from_numpy(xray_b), [0.5], [0.5])

            ct = resize(f[self.ct_path_samples[idx]], (1, 128, 128, 128))
            ct = (ct - np.min(ct))/ (np.max(ct)-np.min(ct))
            ct_array = F.normalize(torch.from_numpy(ct), p=2, dim=2)
            return (torch.cat([xray_a, xray_b], 0), ct)
            

        else: return self.samples[idx]

def projection(m, axis):
    proj = torch.sum(m,axis)
    return proj

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    data_root = '/work/scratch/lan/datasets/hdf5'
    # ct_root = '/work/scratch/lan/datasets/LIDC'
    transform = transforms.Compose([transforms.Normalize([0.5], [0.5])])
    dataset = Dataset_Hdf5(data_root, mode='debug', cache=True)
    data = DataLoader(dataset, batch_size = 1)
    # print(len(dataset))
    # print(dataset[0]['X'])
    # print(dataset[0]['CT'])
    for item in data:
        # print(item[0])
        # print(item[0].shape)
        # print(item[1])
        # print(item[1].shape)
        (x, ct) = item
        
        print(x.shape)
        print(ct.shape)
    # print(dataset[1]['X'].shape)
    # print(dataset[1]['CT'].shape)
    # return dataset
    # print(dataset["LIDC-IDRI-0566"])
    # print(len(dataset))
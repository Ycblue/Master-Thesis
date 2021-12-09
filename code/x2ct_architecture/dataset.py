from torch.utils.data import Dataset
import os
import sys
import torch
import numpy as np
import SimpleITK as sitk
from skimage.transform import resize
from tqdm import tqdm
import torchvision.transforms as transforms
import torch.nn.functional as f


class Dataset(Dataset):
    """Xray and Ct dataset as dictionary.

    Returns paired Xray images under 'X' and another CT image under 'CT'.
    
    Resizes to 128x128 or 128x128x128.

    With `cache=True` all data is written to CPU RAM first, instead of loading each time a dataset is called.

    Example: 

    ```
    dataset = Dataset(xray_root, ct_root, mode='debug', cache=True)
    data = DataLoader(dataset, batch_size = 1)
    for item in data:
        (x, ct) = item
    ```

    """
    

    def __init__(self, xray_root, ct_root, mode, transforms_=None, cache=True):

        xray_root = os.path.join(xray_root, mode)
        ct_root = os.path.join(ct_root, mode)

        # self.transform = transforms.Compose(transforms_)
        self.transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize([0.5], [0.5])
        ])
        self.samples = [] # holds cached images
        self.samples_path = [] # holds path to images when cache=False
        self.cache = cache

        # Xray
        self.xray_samples = [] #list of xray arrays 
        self.x_len = 0
        cases = list(os.scandir(xray_root))
        
        #traverses through path to find dicom files. 
        for case_entry in tqdm(cases):
            case_entry_path = os.path.join(xray_root, case_entry)
            dicoms = os.scandir(case_entry_path)
            temp_list = list(map(lambda x: os.path.join(case_entry_path, x), dicoms)) # map: applies function to each item in iterable list 
            temp_list.reverse()
            if self.cache:
                self.xray_samples.append(xray_dcm_to_array(temp_list))
            self.samples_path.append(temp_list)

        self.x_len = len(self.samples_path)

        # CT
        self.ct_samples = [] #list of ct arrays
        # traverse root to find folder with most dicom files and save that path (other folders exist with other ct scans.)
        cases = list(os.scandir(ct_root))
        for case_entry in tqdm(cases):
            case_entry_path = os.path.join(ct_root, case_entry)
            image_path = ""
            max_count = 0
            max_count_dir = []
            dir1 = os.scandir(case_entry_path)
            for dir1_entry in dir1:
                dir1_entry_path = os.path.join(case_entry_path, dir1_entry)
                dir2 = os.scandir(dir1_entry_path)

                for dir2_entry in dir2:
                    dir2_entry_path = os.path.join(dir1_entry_path, dir2_entry)
                    file_list = os.listdir(dir2_entry_path)
                    file_count = len(file_list)

                    if file_count > max_count:
                        max_count_dir = dir2_entry_path
                        max_count = file_count
            if self.cache:
                self.ct_samples.append(ct_dcm_to_array(max_count_dir))
            self.samples_path.append(max_count_dir)
        
        # save to self.samples if cache
        if self.cache:
            temp_dict = {}
            for x, c in zip(self.xray_samples, self.ct_samples):
                self.samples.append((x, c))


    def __len__(self):
        if self.cache:
            return len(self.samples)
        else:
            return len(self.samples_path)

    def __getitem__(self, idx):
        '''
        Returns sets of images for idx in samples_path or samples, depending on value of cache.

        '''
        if not self.cache:
            output = {}
            output = ( xray_dcm_to_array(self.samples_path[idx]),  ct_dcm_to_array(self.samples_path[idx+self.x_len]) )
            # output['X'] = xray_dcm_to_array(self.samples_path[idx]) 
            # output['CT'] = ct_dcm_to_array(self.samples_path[idx+self.x_len]) 
            return output
        else: return self.samples[idx]

def transform_2d(x):
    transform = transforms.Compose([
                # transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])])
    transformed_x = transform(x)
    return transformed_x

def xray_dcm_to_array(xray_dir):

    # loads xray images from given given path `xray_dir`
    # directory contains three xray images from three different perspectives, but only two of them are needed.
    # returns concat of two resized perspectives as tensor
    a_dir = xray_dir[1]
    b_dir = xray_dir[2]
    
    reader = sitk.ImageFileReader()

    reader.SetFileName(a_dir)
    image_a = reader.Execute()
    array_a = sitk.GetArrayFromImage(image_a)
    array_a = resize(array_a, (1, 128, 128)).astype(np.float32)
    
    reader.SetFileName(b_dir)
    image_b = reader.Execute()
    array_b = sitk.GetArrayFromImage(image_b)
    array_b = resize(array_b, (1, 128, 128)).astype(np.float32)

    array_a = transform_2d(torch.from_numpy(array_a))
    array_b = transform_2d(torch.from_numpy(array_b))
    x_out = torch.cat([array_a, array_b], 0)

    return x_out
    
def ct_dcm_to_array(ct_dir):

    # loads dicom series from given path `ct_dir`
    # returns resized tensor

    data_directory = ct_dir
    series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(data_directory)
    if not series_IDs:
        print("ERROR: given directory \""+data_directory+"\" does not contain a DICOM    series.")
        sys.exit(1)
    series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(data_directory,             series_IDs[0])
    series_reader = sitk.ImageSeriesReader()
    series_reader.SetFileNames(series_file_names)

    series_reader.MetaDataDictionaryArrayUpdateOn()
    series_reader.LoadPrivateTagsOn()
    image3D = series_reader.Execute()
    
    img = sitk.GetArrayFromImage(image3D)
    img = resize(img, (1, 128, 128, 128))
    # img = resize(img, (128, 128, 128))
    # ct_out = transform(img)
    ct_out = f.normalize(torch.from_numpy(img.astype(np.float32)))

    return ct_out

    

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    xray_root = '/work/scratch/lan/datasets/xrays'
    ct_root = '/work/scratch/lan/datasets/LIDC'
    dataset = Dataset(xray_root, ct_root, mode='debug', cache=True)
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
        print('x: ', x.shape)
        print('ct: ', ct.shape)
    # print(dataset[1]['X'].shape)
    # print(dataset[1]['CT'].shape)
    # return dataset
    # print(dataset["LIDC-IDRI-0566"])
    # print(len(dataset))
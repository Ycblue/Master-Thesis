import h5py
import numpy as np
import os
import sys
import torch
import SimpleITK as sitk
# from skimage.transform import resize
from tqdm import tqdm

"""
Saves our xray/ct dataset as numpy arrays (float32) into hdf5 format. 
The output dataset hierarchy will look like this: 

x2ct_dataset.hdf5
    - xrays
        - train
            - LIDC-IDRI-0001 
                - 0 (dataset containing xray projection from one direction)
                - 1
                - 2
        - val
        - test
        - debug (contains 2 datasets)
    - ct
        - train 
            - LIDC-IDRI-0001 (dataset containing ct array)
        - val
        - test
        - debug
"""
def xray_dcm_to_array(xray_dir):

    # loads xray images from given given path `xray_dir`
    # returns numpy array [float32]
    a_dir = xray_dir
    reader = sitk.ImageFileReader()
    reader.SetFileName(a_dir)
    image_a = reader.Execute()
    array_a = sitk.GetArrayFromImage(image_a)
    x_out = array_a.astype(np.float32)

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
    # img = resize(img, (1, 128, 128, 128))
    ct_out = img.astype(np.float32)

    return ct_out

def find_ct_folder(case_path):
    max_count = 0
    max_count_dir = []
    dir1 = os.scandir(case_path)
    for dir1_entry in dir1:
        dir1_entry_path = os.path.join(case_path, dir1_entry)
        dir2 = os.scandir(dir1_entry_path)
        for dir2_entry in dir2:
            dir2_entry_path = os.path.join(dir1_entry_path, dir2_entry)
            file_list = os.listdir(dir2_entry_path)
            file_count = len(file_list)

            if file_count > max_count:
                max_count_dir = dir2_entry_path
                max_count = file_count 
    return max_count_dir

if __name__ == '__main__':
    f = h5py.File('/work/scratch/lan/datasets/hdf5/x2ct_dataset.hdf5', 'w') #file object
    xray_grp = f.create_group('xrays')
    ct_grp = f.create_group('LIDC')
    modes = ['train', 'val', 'test', 'debug']
    # modes = ['debug']
    xray_root = '/work/scratch/lan/datasets/xrays'
    ct_root = '/work/scratch/lan/datasets/LIDC'

    for m in modes:
        # xray
        x_m_grp = xray_grp.create_group('{}'.format(m))
        ct_m_grp = ct_grp.create_group('{}'.format(m))
        path = os.path.join(xray_root, m)
        for case in tqdm(list(os.scandir(path))):
            case_path = os.path.join(path, case)
            dicoms = os.scandir(case)
            # grp = x_m_grp.create_group('{}'.format(case))
            temp_list = list(map(lambda x: xray_dcm_to_array(os.path.join(case_path, x)), dicoms))
            for i in range(len(temp_list)):
                dataset = x_m_grp.create_dataset('{}/{}'.format(case, i), data=temp_list[i])
        # ct
        path = os.path.join(ct_root, m)
        for case in tqdm(list(os.scandir(path))):
            case_path = os.path.join(path, case)

            # grp = ct_m_grp.create_group('{}'.format(case))
            max_count = 0
            max_count_dir = []
            dir1 = os.scandir(case_path)
            for dir1_entry in dir1:
                dir1_entry_path = os.path.join(case_path, dir1_entry)
                dir2 = os.scandir(dir1_entry_path)
                for dir2_entry in dir2:
                    dir2_entry_path = os.path.join(dir1_entry_path, dir2_entry)
                    file_list = os.listdir(dir2_entry_path)
                    file_count = len(file_list)

                    if file_count > max_count:
                        max_count_dir = dir2_entry_path
                        max_count = file_count 
            
            # dataset = ct_m_grp.create_dataset('{}'.format(case), data=ct_dcm_to_array(max_count_dir))
            dataset = ct_m_grp.create_dataset('{}'.format(case), data=ct_dcm_to_array(find_ct_folder(case_path)))
    print('Done.')
    # for key in xray_grp.keys():
    #     for key_2 in xray_grp[key]:
    #         print(key_2)
    #         print(xray_grp[key][key_2])
    #     # print(xray_grp[key])
    # # print(xray_grp.items())
    # print(f['xrays/debug'].keys())
    
import os
import tempfile
import pydicom 
from pydicom.dataset import Dataset, FileDataset
import matplotlib.pyplot as plt

import sys
import glob
import numpy as np

import SimpleITK as sitk
from ct2x_projection import do_projection
from ct2x_projection import project_series
from skimage.transform import resize
import math

#WIP read dicom series 
# file_name = '/work/scratch/lan/datasets/LIDC/LIDC-IDRI-0001/01-01-2000-30178/3000566-03192/000001.dcm'
data_directory = '/work/scratch/lan/datasets/LIDC/LIDC-IDRI-0001/01-01-2000-30178/3000566-03192'
output_directory = 'output'

series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(data_directory)
if not series_IDs:
   print("ERROR: given directory \""+data_directory+"\" does not contain a DICOM    series.")
   sys.exit(1)
series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(data_directory,             series_IDs[0])
# Read the file's meta-information without reading bulk pixel data
series_reader = sitk.ImageSeriesReader()
series_reader.SetFileNames(series_file_names)

series_reader.MetaDataDictionaryArrayUpdateOn()
series_reader.LoadPrivateTagsOn()
image3D = series_reader.Execute()

voxel_dimensions = np.asarray(image3D.GetSpacing()) 
voxel_dimensions = [i*0.001 for i in voxel_dimensions]
voxel_dimensions.reverse()
print(voxel_dimensions)

n_img = sitk.GetArrayFromImage(image3D)

for i in range(3):

    axis = i
    print(axis)
    print(voxel_dimensions[axis])
    xray = do_projection(n_img, axis, voxel_dimensions[axis])
    print('Projection Done')
    #resize
    xray = resize(xray, (512, 512))

    #window
    x_max = np.max(a=xray)
    x_min = np.min(a=xray)
    xray = (xray - x_min) * (65535/(x_max-x_min))

    #type
    xray_uint16 = xray.astype(np.uint16)

    #clip
    x_max = np.max(a=xray_uint16)
    p = 0.1
    cutoff = x_max * p
    xray_uint16 = np.clip(xray, cutoff, 65535)


    # _ = plt.hist(xray_uint16.reshape(-1).astype(np.int64), bins='auto')  # arguments are passed to np.histogram
    # plt.title("Histogram with 'auto' bins")
    # plt.show()

    #invert
    xray_uint16 = 65535 - xray_uint16
    #rotate
    if axis > 0:
        xray_uint16 = np.rot90(xray_uint16, 2)
        xray_uint16 = np.flip(xray_uint16, 1)
    
    #save
    img = sitk.GetImageFromArray(xray_uint16.astype(np.uint16))
    # sitk.WriteImage(img, '%s/raw_projection%d.dcm' % (output_directory, i))

project_series(data_directory, output_directory)

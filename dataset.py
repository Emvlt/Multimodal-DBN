#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 11:00:26 2019

@author: emmv1d18
"""
import torch
import numpy as np
import pickle
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class dataset(Dataset):
    def __init__(self, data_path, data_type, transform=None):
        """
        Args:
        data_path (string) : Path to the dataset folder
        data_type (string) : type of dataset 'train' or 'test'
        """
        self.data_path = data_path
        self.transform = transform
        self.masks = np.asarray(pickle.load(open(data_path+data_type+'_'+'masks','rb')))
        self.cads  = np.asarray(pickle.load(open(data_path+data_type+'_'+'cads','rb')))
        self.objects_target     = np.asarray(pickle.load(open(data_path+data_type+'_'+'objects_target','rb')))
        self.objects_corrupted  = np.asarray(pickle.load(open(data_path+data_type+'_'+'objects_corrupted','rb')))


    def __len__(self):
        return len(self.objects_target)

    def __getitem__(self, idx):
        sinogram_object_corrupted = self.objects_corrupted[idx]
        sinogram_object_target    = self.objects_target[idx]
        sinogram_cad  = self.cads[idx]
        sinogram_mask = self.masks[idx]
        sample = {'obj_cor': sinogram_object_corrupted, 'obj_tgt': sinogram_object_target,'cad': sinogram_cad, 'mask':sinogram_mask}
        if self.transform:
            sample = self.transform(sample)
        return sample

class Normalize(object):
    """Normalize a sample of dataset so that it has a zero mean and std_dev of 1"""
    def __call__(self, sample):
        sinogram_object_target, sinogram_object_corrupted = sample['obj_tgt'], sample['obj_cor']
        sinogram_cad, sinogram_mask =  sample['cad'], sample['mask']
        sinogram_object_corrupted = (sinogram_object_corrupted - sinogram_object_corrupted.mean())/sinogram_object_corrupted.std()
        sinogram_object_target    = (sinogram_object_target - sinogram_object_target.mean())/sinogram_object_target.std()
        sinogram_cad    = (sinogram_cad - sinogram_cad.mean())/sinogram_cad.std()
        return {'obj_cor': sinogram_object_corrupted, 'obj_tgt': sinogram_object_target,'cad': sinogram_cad, 'mask':sinogram_mask}

class Corrupt_frames(object):
    """Corrupts n_frames columns in the image by replacing the initial values by 0"""  
    def __init__(self, frames):
        self.n_frames = frames
    def __call__(self, sample):
        sinogram_object, sinogram_cad = sample['object'], sample['cad']
        circle_object, circle_cad = sample['circle_object'], sample['circle_cad']
        sinogram_object = (sinogram_object - sinogram_object.mean())/sinogram_object.std()
        sinogram_cad    = (sinogram_cad - sinogram_cad.mean())/sinogram_cad.std()
        sinogram_object_ref = sinogram_object.copy()
        number_of_missing_frames = np.random.randint(self.n_frames)
        case = random.sample(range(256), number_of_missing_frames)
        #case = random.sample(range(256), self.n_frames)
        sinogram_object[:,:,[i for i in case]] =0
        # Create the Importance Weighted Context Term
        # Create the Mask
        M = np.ones((1,1,256,256))
        M[0,0,:,case] = 0
        M = torch.from_numpy(M).float()
        # Create the 1 matrix
        One = torch.ones((1,1,256,256)).float()
        # Create the sum filter
        sum_kernel = torch.ones((1,1,7,7)).float()
        # Create W
        W = torch.nn.functional.conv2d(One-M, sum_kernel, padding=3)/49
        W = W*M

        return {
            'object': sinogram_object,
            'object_reference' : sinogram_object_ref,
            'cad'   : sinogram_cad,
            'list'  : case,
            'W'     : W.reshape((1,256,256)),
            'circle_object' : circle_object,
            'circle_cad' : circle_cad
        }

class Corrupt_chunks(object):
    def __call__(self, sample):
        sinogram_object, sinogram_cad = sample['object'], sample['cad']
        circle_object, circle_cad = sample['circle_object'], sample['circle_cad']
        sinogram_object = (sinogram_object - sinogram_object.mean())/sinogram_object.std()
        sinogram_cad    = (sinogram_cad - sinogram_cad.mean())/sinogram_cad.std()
        sinogram_object_ref = sinogram_object.copy()
        location = np.random.randint(256)
        size     = np.random.randint(128)
        sinogram_object[:,:,location-size:location+size] =0
        # Create the Importance Weighted Context Term
        # Create the Mask
        M = np.ones((1,1,256,256))
        M[0,0,:,location-size:location+size] = 0
        M = torch.from_numpy(M).float()
        # Create the 1 matrix
        One = torch.ones((1,1,256,256)).float()
        # Create the sum filter
        sum_kernel = torch.ones((1,1,7,7)).float()
        # Create W
        W = torch.nn.functional.conv2d(One-M, sum_kernel, padding=3)/49
        W = W*M

        return {
            'object': sinogram_object,
            'object_reference' : sinogram_object_ref,
            'cad'   : sinogram_cad,
            'list'  : (location,size),
            'W'     : W.reshape((1,256,256)),
            'circle_object' : circle_object,
            'circle_cad' : circle_cad
        }

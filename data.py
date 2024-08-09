#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import torch
import torchvision
from torch.utils.data import Dataset 
from torchvision.datasets import VisionDataset, ImageFolder
from PIL import Image
from typing import Any, Callable, Optional, Tuple
from torch.distributions.beta import Beta

import pathlib
import os
import numpy as np
import glob


class WeightedDataset(Dataset):
    '''
    Extends Dataset to include weights
    '''
    def __init__(self, data, labels, weights, transform=None, is_image=True):
        self.data = data
        self.labels = labels
        self.weights = weights
        self.transform = transform
        self.is_image = is_image
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        w = self.weights[idx]
  
        return (x, y, w)
    

# developed upon https://github.com/pytorch/vision/blob/main/torchvision/datasets/cifar.py
class WeightedImageDataset(VisionDataset):
    '''
    Extends VisionDataset to include weights
    '''
    def __init__(self, data, labels, weights, transform=None, is_image=True):
        self.data = data
        self.labels = labels
        self.weights = weights
        self.transform = transform
        
    @property 
    def targets(self):
        return self.labels 
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target, weight = self.data[index], self.labels[index], self.weights[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target, weight 


class ImageDataset(VisionDataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
        
    @property 
    def targets(self):
        return self.labels
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.labels[index] 
        
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target  
    

class BalancedImageDataset(ImageDataset):
    def __init__(self, data, labels, data_0, data_1, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
        # consider a binary class here, data_0 and data_1 already shuffled
        self.data_class0 = data_0 
        self.data_class1 = data_1
        self.n0 = len(data_0)
        self.n1 = len(data_1)
        
    @property 
    def targets(self):
        return self.labels
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.labels[index] 
        
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        img0 = Image.fromarray(self.data_class0[index%self.n0])
        img1 = Image.fromarray(self.data_class1[index%self.n1])

        if self.transform is not None:
            img = self.transform(img)
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img, target, img0, img1  


DATASET_NUM_CLASSES = {"cifar10_1": 10, 
                       "cinic10": 10, 
                       "cifar10_c": 10, 
                       "cifar10": 10, 
                       "cifar100": 100, 
                       "cifar100_c": 100,
                       "kmnist": 10}



def get_dataset(name: str, **kwargs):
    if name == 'cifar10':
        # test data 
        dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, **kwargs)
    if name == 'cifar10_1':
        images, labels = load_cifar10_1()
        dataset = ImageDataset(data=images, labels=torch.tensor(labels, dtype=torch.long), **kwargs)
    elif name == 'cinic10':
        # imagenet data only 
        data_directory = os.path.join(os.path.dirname(__file__), './data/CINIC-10/test')  
        dataset = ImageFolder(data_directory, **kwargs) 
    elif  name[:len('cifar10_c')] == 'cifar10_c':
        # name format "cifar10_c_{corruption}"
        # load all 5 corruption levels 
        corruption = name[len('cifar10_c_'):]
        name = 'cifar10_c'
        data_path = os.path.join(os.path.dirname(__file__), f'./data/CIFAR-10-C/{corruption}.npy')
        label_path = os.path.join(os.path.dirname(__file__), './data/CIFAR-10-C/labels.npy') 
        
        images = np.load(data_path)
        labels = np.load(label_path)
        dataset = ImageDataset(data=images, labels=torch.tensor(labels, dtype=torch.long), **kwargs)
    elif name[:len('cifar100_c')] == 'cifar100_c':
        # name format "cifar100_c_{corruption}"
        # load all 5 corruption levels 
        corruption = name[len('cifar100_c_'):]
        name = 'cifar100_c'
        data_path = os.path.join(os.path.dirname(__file__), f'./data/CIFAR-100-C/{corruption}.npy')
        label_path = os.path.join(os.path.dirname(__file__), './data/CIFAR-100-C/labels.npy') 
        
        images = np.load(data_path)
        labels = np.load(label_path)
        dataset = ImageDataset(data=images, labels=torch.tensor(labels, dtype=torch.long), **kwargs)
    elif name == 'kmnist':            
        dataset = torchvision.datasets.KMNIST(root='./data', train=False, download=True, **kwargs)
    else:
        raise NotImplementedError(f"Dataset {name} is not implemented.")
    return dataset, DATASET_NUM_CLASSES[name]
                    

# reference: https://github.com/modestyachts/CIFAR-10.1/blob/master/notebooks/inspect_dataset_simple.ipynb 
def load_cifar10_1(verbose=False):
    data_path = os.path.join(os.path.dirname(__file__), './data/')
    filename = 'cifar10.1'

    version_string = 'v6'
    filename += '_' + version_string
    label_filename = filename + '_labels.npy'
    imagedata_filename = filename + '_data.npy'
    label_filepath = os.path.abspath(os.path.join(data_path, label_filename))
    imagedata_filepath = os.path.abspath(os.path.join(data_path, imagedata_filename))
    if verbose:
        print('Loading labels from file {}'.format(label_filepath))
    assert pathlib.Path(label_filepath).is_file()
    labels = np.load(label_filepath)
    if verbose:
        print('Loading image data from file {}'.format(imagedata_filepath))
    assert pathlib.Path(imagedata_filepath).is_file()
    imagedata = np.load(imagedata_filepath)
    assert len(labels.shape) == 1
    assert len(imagedata.shape) == 4
    assert labels.shape[0] == imagedata.shape[0]
    assert imagedata.shape[1] == 32
    assert imagedata.shape[2] == 32
    assert imagedata.shape[3] == 3
    if version_string == 'v6' or version_string == 'v7':
        assert labels.shape[0] == 2000
    elif version_string == 'v4':
        assert labels.shape[0] == 2021

    return imagedata, list(labels)


__DATA_AUG__ = {} 


def register_data_aug(name: str): 
    def wrapper(cls):
        if __DATA_AUG__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __DATA_AUG__[name] = cls 
    return wrapper


def get_data_aug(name: str, **kwargs):
    if __DATA_AUG__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined!")
    return __DATA_AUG__[name](**kwargs)


@register_data_aug('mixupv0')
def mixupv0(x, y, a=1.0):
    interp  = Beta(a,a).sample().to(x.device)
    indices = torch.randperm(x.shape[0], device=x.device)
    mixed_x = interp * x + (1-interp)*x[indices]

    y_ind = torch.bernoulli(interp*torch.ones(y.shape[0], device=x.device))
    mixed_y = y_ind * y + (1-y_ind) * y[indices]
    mixed_y = mixed_y.type(y.dtype)
    return mixed_x, mixed_y 


@register_data_aug('mixupv1')
def mixupv1(x, y, a=1.0):
    interp  = Beta(a,a).sample((x.shape[0],)).to(x.device)
    indices = torch.randperm(x.shape[0], device=x.device)
    
    interp_ = interp
    for _ in range(len(x.shape)-1):
        interp_ = interp_.unsqueeze(-1)
    mixed_x = interp_ * x + (1-interp_)*x[indices]

    y_ind = torch.bernoulli(interp*torch.ones(y.shape[0], device=y.device))
    mixed_y = y_ind * y + (1-y_ind) * y[indices]
    mixed_y = mixed_y.type(y.dtype)
    return mixed_x, mixed_y 


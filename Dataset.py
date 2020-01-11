#!/usr/bin/python
# -*- coding: UTF-8 -*-
###################### load packages ########################
import torch.utils.data
import pandas as pd
from skimage import io
import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"

###################### Dataset class ########################
class ImageDataset(torch.utils.data.Dataset):

    ############ init ###########
    def __init__(self, filenames, train=True, transform=None):
        self.transform = transform
        self.train = train

        self.train_data = pd.read_csv(filenames, sep=',', header=None)
        self.test_data = pd.read_csv(filenames, sep=',', header=None)

        self.stream = torch.cuda.Stream()


    ############ get data ###########
    def __getitem__(self, index):

        if self.train:
            fn, label = self.train_data.iloc[index:index+1, 0].values[0], self.train_data.iloc[index:index+1, 1].values[0]
        else:
            fn, label = self.test_data.iloc[index:index + 1, 0].values[0], self.test_data.iloc[index:index + 1, 1].values[0]

        img = io.imread('/share/home/Chenxi/SQ' + fn)
        if img.dtype!="float32":
            print(fn, img.dtype)

        torch.cuda.current_stream().wait_stream(self.stream)

        with torch.cuda.stream(self.stream):
            img = torch.Tensor(img.astype('float32')).cuda()
            label = torch.tensor(int(label)).cuda()

        return img, label

    ############ get data length ###########
    def __len__(self):
        if self.train:
           return len(self.train_data)
        else:
           return len(self.test_data)
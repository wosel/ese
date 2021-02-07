import torch

import torch.nn as nn
import numpy as np
from torchvision import datasets, models, transforms

from argparse import ArgumentParser

import sys
from cv2 import imread

from dataset import TrainCouplingDataset


if __name__ == '__main__':
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_loaded = models.resnet18(pretrained=True)
    #set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_loaded.fc.in_features
    model_loaded.fc = nn.Linear(num_ftrs, 2)
    model_loaded.load_state_dict(torch.load('model.pth'))
    model_loaded.eval()
    model_loaded.to(device)

    imagepaths = sys.argv[1:]

    print(imagepaths)

    dset = TrainCouplingDataset(imagepaths, './', mode='test')
    
    
    for im_idx in range(len(dset)):
        orig  = imread(imagepaths[im_idx])
        res = model_loaded(dset[im_idx]['image'].unsqueeze(0).to(device)).detach().cpu().numpy()
        mid = int((res[0][0] + res[0][1])/2 * orig.shape[1])
        print(mid)
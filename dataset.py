from torch.utils.data import Dataset
import torch
import numpy as np
import cv2
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage
import os
import json 


class TrainCouplingDataset(Dataset):
    
    def __init__(self, filename_list, root_dir, mode='val'):
        self.filename_list = filename_list
        self.root_dir = root_dir
        self.mode=mode
        self.train_seq = iaa.Sequential([
            iaa.Resize(448),
            iaa.Sometimes(
                0.2,
                iaa.MotionBlur(k=(5, 15)),
            ),
            iaa.Add((-40, 40)),
            iaa.Multiply((1.2, 1.5)), # change brightness, doesn't affect keypoints
            iaa.Fliplr(0.5),
            iaa.Sometimes(
                0.2,
                iaa.GaussianBlur(sigma=(1.0, 5.0)),
            ),
            
        ])
        self.val_seq = iaa.Sequential([
            iaa.Resize(448),
            
        ])
 
    def __len__(self):
        return len(self.filename_list)

    def __getitem__(self, idx):
        try:

            
            image = cv2.imread(self.filename_list[idx])

            if self.mode != 'test':
                img_fname = self.filename_list[idx]
                json_fname = '.'.join(img_fname.split('.')[:-1])+'.json'
                j = json.load(open(json_fname))
                pts = j['shapes'][0]['points']
                
            if self.mode == 'train':
                seq = self.train_seq
            else:
                seq = self.val_seq
            
            if self.mode != 'test':
                kps = KeypointsOnImage([
                    Keypoint(x=pt[0], y=pt[1]) for pt in pts], 
                    shape = image.shape
                )
            else:
                kps = []
            image_aug, kps_aug = seq(image=image, keypoints=kps)
            
            if self.mode != 'test':
                minx =  min([pt.x for pt in kps_aug])
                maxx = max([pt.x for pt in kps_aug])
                target = [minx/image_aug.shape[1], maxx/image_aug.shape[1]]
            
                sample = {'image': torch.tensor(image_aug).permute(2, 0, 1).type(torch.FloatTensor), 'bbox': torch.tensor(target).type(torch.FloatTensor)}
            else:
                sample = {'image': torch.tensor(image_aug).permute(2, 0, 1).type(torch.FloatTensor)}
            return sample
        except Exception as e:
            print(e)
            raise
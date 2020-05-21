import os, sys
sys.path.append('.')

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

import config
class FaceDataset(Dataset):
    '''Dataset class for MTCNN face detector'''
    def __init__(self, annotation_path):
        with open(annotation_path, 'r') as f:
            self.img_files = f.readlines()
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, index):
        annotation = self.img_files[index % len(self.img_files)].strip().split(' ')
        img = cv2.imread(annotation[0], 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.asarray(img, 'float32')
        img = np.transpose(img, (2, 0, 1))
        img = (img - 127.5) / 127.5 # rescale pixel value to between -1 and 1
        input_img = torch.FloatTensor(img)

        cls_target = int(annotation[1])
        bbox_target = np.zeros((4,))
        landmark_target = np.zeros((10,))

        if len(annotation[2:]) == 4:
            bbox_target = np.array(annotation[2:6]).astype(float)
        if len(annotation[2:]) == 14:
            bbox_target = np.array(annotation[2:6]).astype(float)
            landmark_target = np.array(annotation[2:6]).astype(float)
        sample = {'input_img': input_img, 
                  'cls_target': cls_target, 
                  'bbox_target': bbox_target,
                  'landmark_target': landmark_target}
        return sample

if __name__ == '__main__':
    
    
    
    train_data = Dataset(os.path.join(config.ANNO_PATH, config.PNET_TRAIN_IMGLIST_FILENAME))
    val_data = Dataset(os.path.join(config.ANNO_PATH, config.PNET_VAL_IMGLIST_FILENAME))
    dataloaders = {'train': torch.utils.data.DataLoader(train_data, 
                                batch_size=config.BATCH_SIZE, shuffle=True),
                    'val': torch.utils.data.DataLoader(val_data,
                                batch_size=config.BATCH_SIZE, shuffle=True)
                  }

    for batch_idx, sample_batched in enumerate(dataloaders['train']):
        images_batch, cls_batch, bbox_batch, landmark_batch = \
                sample_batched['input_img'], sample_batched['cls_target'], sample_batched['bbox_target'], sample_batched['landmark_target']  
                                                                                                 
        print(batch_idx, images_batch.shape, cls_batch.shape, bbox_batch.shape, landmark_batch.shape)
        
        if batch_idx == 3:
            break       




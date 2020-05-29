import sys
sys.path.append('./')
import os
import argparse

import torch
from torchvision import transforms

from tools.dataset import FaceDataset
from nets.mtcnn import ONet
from training.onet.landmark_trainer import ONetTrainer
from checkpoint import CheckPoint
import config

# Set device
use_cuda = config.USE_CUDA and torch.cuda.is_available()
device = torch.device("cuda:1" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

# Set dataloader
kwargs = {'num_workers': 8, 'pin_memory': True} if use_cuda else {}
train_data = FaceDataset(os.path.join(config.ANNO_PATH, config.ONET_TRAIN_IMGLIST_FILENAME))
val_data = FaceDataset(os.path.join(config.ANNO_PATH, config.ONET_VAL_IMGLIST_FILENAME))
dataloaders = {'train': torch.utils.data.DataLoader(train_data, 
                        batch_size=config.BATCH_SIZE, shuffle=True, **kwargs),
               'val': torch.utils.data.DataLoader(val_data,
                        batch_size=config.BATCH_SIZE, shuffle=True, **kwargs)
              }

# Set model
model = ONet(is_train=True, train_landmarks=True)
model = model.to(device)
model.load_state_dict(torch.load('pretrained_weights/mtcnn/best_onet.pth'), strict=True)
print(model)

# Set checkpoint
#checkpoint = CheckPoint(train_config.save_path)

# Set optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.STEPS, gamma=0.1)

# Set trainer
trainer = ONetTrainer(config.EPOCHS, dataloaders, model, optimizer, scheduler, device)

trainer.train()
    
#checkpoint.save_model(model, index=epoch, tag=config.SAVE_PREFIX)
            

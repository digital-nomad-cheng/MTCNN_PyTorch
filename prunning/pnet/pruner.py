import os, sys
sys.path.append('.')
import time
import datetime
from collections import OrderedDict
import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from loss import Loss
from tools.average_meter import AverageMeter

class PNetPruner(object):
    
    def __init__(self, epochs, dataloaders, model, optimizer, scheduler, device,
            prune_ratio, finetune_epochs):
        self.epochs = epochs
        self.dataloaders = dataloaders
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.lossfn = Loss(self.device)
        
        self.prune_iters = self._estimate_pruning_iterations(model, prune_ratio)
        print("Total prunning iterations:", self.prune_iters)
        self.finetune_epochs = finetune_epochs

    def compute_accuracy(self, prob_cls, gt_cls):
        # we only need the detection which >= 0
        prob_cls = torch.squeeze(prob_cls)
        mask = torch.ge(gt_cls, 0)
        
        # get valid elements
        valid_gt_cls = gt_cls[mask]
        valid_prob_cls = prob_cls[mask]
        size = min(valid_gt_cls.size()[0], valid_prob_cls.size()[0])
        
        # get max index with softmax layer
        _, valid_pred_cls = torch.max(valid_prob_cls, dim=1)
        
        right_ones = torch.eq(valid_pred_cls.float(), valid_gt_cls.float()).float()

        return torch.div(torch.mul(torch.sum(right_ones), float(1.0)), float(size))
    
    def prune(self):
        print("Before Prunning...")
        self.train_epoch(0, 'val')
        for i in range(self.prune_iters):
            self.prune_step()
            print("After Prunning Iter ", i)
            self.train_epoch(i, 'val')
            print("Finetuning...")
            for epoch in range(self.finetune_epochs):
                self.train_epoch(i, 'train') 
                self.train_epoch(i, 'val')
            torch.save(self.model.state_dict(), './prunning/results/pruned_pnet.pth')
            torch.onnx.export(self.model, torch.randn(1, 3, 12, 12).to(self.device), 
                    './onnx2ncnn/pruned_pnet.onnx',
                    input_names=['input'], output_names=['scores', 'offsets'])
    
    def prune_step(self):
        self.model.train()
        
        sample_idx = np.random.randint(0, len(self.dataloaders['train']))
        for batch_idx, sample in enumerate(self.dataloaders['train']):
            if batch_idx == sample_idx:
                data = sample['input_img']
                gt_cls = sample['cls_target']
                gt_bbox = sample['bbox_target']
        
        data, gt_cls, gt_bbox = data.to(self.device), gt_cls.to(self.device), gt_bbox.to(self.device).float()
        pred_cls, pred_bbox = self.model(data)
        cls_loss = self.lossfn.cls_loss(gt_cls, pred_cls)
        bbox_loss = self.lossfn.box_loss(gt_cls, gt_bbox, pred_bbox)
        total_loss = cls_loss + 5*bbox_loss
        total_loss.backward()
        self.model.prune(self.device)


    def train_epoch(self, epoch, phase):
        cls_loss_ = AverageMeter()
        bbox_loss_ = AverageMeter()
        total_loss_ = AverageMeter()
        accuracy_ = AverageMeter()
        if phase == 'train':
            self.model.train()
        else:
            self.model.eval()

        for batch_idx, sample in enumerate(self.dataloaders[phase]):
            data = sample['input_img']
            gt_cls = sample['cls_target']
            gt_bbox = sample['bbox_target']
            data, gt_cls, gt_bbox = data.to(self.device), gt_cls.to(
                self.device), gt_bbox.to(self.device).float()
            
            self.optimizer.zero_grad()
            with torch.set_grad_enabled(phase == 'train'):
                pred_cls, pred_bbox = self.model(data)
                
                # compute the cls loss and bbox loss and weighted them together
                cls_loss = self.lossfn.cls_loss(gt_cls, pred_cls)
                bbox_loss = self.lossfn.box_loss(gt_cls, gt_bbox, pred_bbox)
                total_loss = cls_loss + 5*bbox_loss
                
                # compute clssification accuracy
                accuracy = self.compute_accuracy(pred_cls, gt_cls)

                if phase == 'train':
                    total_loss.backward()
                    self.optimizer.step()

            cls_loss_.update(cls_loss, data.size(0))
            bbox_loss_.update(bbox_loss, data.size(0))
            total_loss_.update(total_loss, data.size(0))
            accuracy_.update(accuracy, data.size(0))
            
            #if batch_idx % 40 == 0:
            #    print('{} Epoch: {} [{:08d}/{:08d} ({:02.0f}%)]\tLoss: {:.6f} cls Loss: {:.6f} offset Loss:{:.6f}\tAccuracy: {:.6f} LR:{:.7f}'.format(
            #        phase, epoch, batch_idx * len(data), len(self.dataloaders[phase].dataset),
            #        100. * batch_idx / len(self.dataloaders[phase]), total_loss.item(), cls_loss.item(), bbox_loss.item(), accuracy.item(), self.optimizer.param_groups[0]['lr']))
        
        print("{} epoch Loss: {:.6f} cls Loss: {:.6f} bbox Loss: {:.6f} Accuracy: {:.6f}".format(
            phase, total_loss_.avg, cls_loss_.avg, bbox_loss_.avg, accuracy_.avg))
        
        # torch.save(self.model.state_dict(), './pretrained_weights/quant_mtcnn/best_pnet.pth')
        
        return cls_loss_.avg, bbox_loss_.avg, total_loss_.avg, accuracy_.avg
   
    
    def _estimate_pruning_iterations(self, model, prune_ratio):
        '''Estimate how many feature maps to prune using estimated params per 
        feature map divide by total param to prune, since we only prune 1 filter
        at a time, iterations should equal to total filters to prune
        
        Parameters:
        -----------
        model: pytorch model
        prune_ratio: ration of total params to prune
        
        Return: 
        -------
        num of iterations of pruning
        '''
        # we only prune Conv2d layers here, Linear layer will be considered later
        conv2ds = [module for module in model.modules() 
                if issubclass(type(module), nn.Conv2d)]
        num_feature_maps = np.sum(conv2d.out_channels for conv2d in conv2ds)
        
        conv2d_params = (module.parameters() for module in model.modules() 
                if issubclass(type(module), nn.Conv2d))
        param_objs = itertools.chain(*conv2d_params)
        # num_param: in * out * w * h per feature map
        num_params = np.sum(np.prod(np.array(p.size())) for p in param_objs)
        
        params_per_map = num_params // num_feature_maps
        
        
        return int(np.ceil(num_params * prune_ratio / params_per_map))

if __name__ == "__main__":
    pass

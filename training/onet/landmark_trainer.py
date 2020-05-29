import time
import datetime

import torch

import config
from loss import Loss
from tools.average_meter import AverageMeter


class ONetTrainer(object):
    
    def __init__(self, epochs, dataloaders, model, optimizer, scheduler, device):
        self.epochs = epochs
        self.dataloaders = dataloaders
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.lossfn = Loss(self.device)
        
        # save best model
        self.best_val_loss = 100

    def compute_accuracy(self, prob_cls, gt_cls):
        # we only need the detection which >= 0
        prob_cls = torch.squeeze(prob_cls)
        tmp_gt_cls = gt_cls.detach().clone()
        tmp_gt_cls[tmp_gt_cls==-2] = 1
        mask = torch.ge(tmp_gt_cls, 0)
        
        # get valid elements
        valid_gt_cls = tmp_gt_cls[mask]
        valid_prob_cls = prob_cls[mask]
        size = min(valid_gt_cls.size()[0], valid_prob_cls.size()[0])
        
        # get max index with softmax layer
        _, valid_pred_cls = torch.max(valid_prob_cls, dim=1)
        
        right_ones = torch.eq(valid_pred_cls.float(), valid_gt_cls.float()).float()

        return torch.div(torch.mul(torch.sum(right_ones), float(1.0)), float(size))
    
    def train(self):
        for epoch in range(self.epochs):
            self.train_epoch(epoch, 'train')
            self.train_epoch(epoch, 'val')

        
    def train_epoch(self, epoch, phase):
        cls_loss_ = AverageMeter()
        bbox_loss_ = AverageMeter()
        landmark_loss_ = AverageMeter()
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
            gt_landmark = sample['landmark_target']
            
            data, gt_cls, gt_bbox, gt_landmark = data.to(self.device), \
                gt_cls.to(self.device), gt_bbox.to(self.device).float(), \
                gt_landmark.to(self.device).float()

            self.optimizer.zero_grad()
            with torch.set_grad_enabled(phase == 'train'):
                pred_cls, pred_bbox, pred_landmark = self.model(data)
                # compute the cls loss and bbox loss and weighted them together
                cls_loss = self.lossfn.cls_loss(gt_cls, pred_cls)
                bbox_loss = self.lossfn.box_loss(gt_cls, gt_bbox, pred_bbox)
                landmark_loss = self.lossfn.landmark_loss(gt_cls, gt_landmark, pred_landmark)
                total_loss = cls_loss + 20*bbox_loss + 20*landmark_loss
                
                # compute clssification accuracy
                accuracy = self.compute_accuracy(pred_cls, gt_cls)

                if phase == 'train':
                    total_loss.backward()
                    self.optimizer.step()

            cls_loss_.update(cls_loss, data.size(0))
            bbox_loss_.update(bbox_loss, data.size(0))
            landmark_loss_.update(landmark_loss, data.size(0))
            total_loss_.update(total_loss, data.size(0))
            accuracy_.update(accuracy, data.size(0))
             
            if batch_idx % 40 == 0:
                print('{} Epoch: {} [{:08d}/{:08d} ({:02.0f}%)]\tLoss: {:.6f} cls Loss: {:.6f} offset Loss:{:.6f} landmark Loss: {:.6f}\tAccuracy: {:.6f} LR:{:.7f}'.format(
                    phase, epoch, batch_idx * len(data), len(self.dataloaders[phase].dataset),
                    100. * batch_idx / len(self.dataloaders[phase]), total_loss.item(), cls_loss.item(), bbox_loss.item(), landmark_loss.item(), accuracy.item(), self.optimizer.param_groups[0]['lr']))
        
        if phase == 'train':
            self.scheduler.step()
        
        print("{} epoch Loss: {:.6f} cls Loss: {:.6f} bbox Loss: {:.6f} landmark Loss: {:.6f} Accuracy: {:.6f}".format(
            phase, total_loss_.avg, cls_loss_.avg, bbox_loss_.avg, landmark_loss_.avg, accuracy_.avg))
        
        if phase == 'val' and total_loss_.avg < self.best_val_loss:
            self.best_val_loss = total_loss_.avg
            torch.save(self.model.state_dict(), './pretrained_weights/mtcnn/best_onet_landmark_2.pth')
        
        return cls_loss_.avg, bbox_loss_.avg, total_loss_.avg, landmark_loss_.avg, accuracy_.avg
    

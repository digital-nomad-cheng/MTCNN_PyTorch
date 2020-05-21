import torch
import torch.nn as nn
import torch.nn.functional as F

class Loss:
    """Losses for classification, face box regression, landmark regression"""
    def __init__(self, device):
        # loss function
        # self.loss_cls = nn.BCELoss().to(device) use this loss for sigmoid score
        self.loss_cls = nn.CrossEntropyLoss().to(device)
        self.loss_box = nn.MSELoss().to(device)
        self.loss_landmark = nn.MSELoss().to(device)


    def cls_loss(self, gt_label, pred_label):
        # get the mask element which >= 0, only 0 and 1 can effect the detection loss
        # kind of confused here, maybe its related to cropped data state
        pred_label = torch.squeeze(pred_label)
        mask = torch.ge(gt_label, 0) # mask is a BoolTensor, select indexes greater or equal than 0
        valid_gt_label = torch.masked_select(gt_label, mask)# .float()
        #valid_pred_label = torch.masked_select(pred_label, mask)
        valid_pred_label = pred_label[mask, :]
        return self.loss_cls(valid_pred_label, valid_gt_label)


    def box_loss(self, gt_label, gt_offset,pred_offset):
        # get the mask element which != 0
        mask = torch.ne(gt_label, 0)
        
        # convert mask to dim index
        chose_index = torch.nonzero(mask)
        chose_index = torch.squeeze(chose_index)
        
        # only valid element can effect the loss
        valid_gt_offset = gt_offset[chose_index,:]
        valid_pred_offset = pred_offset[chose_index,:]
        valid_pred_offset = torch.squeeze(valid_pred_offset)
        return self.loss_box(valid_pred_offset,valid_gt_offset)


    def landmark_loss(self, gt_label, gt_landmark, pred_landmark):
        mask = torch.eq(gt_label,-2)
        
        chose_index = torch.nonzero(mask.data)
        chose_index = torch.squeeze(chose_index)

        valid_gt_landmark = gt_landmark[chose_index, :]
        valid_pred_landmark = pred_landmark[chose_index, :]
        return self.loss_landmark(valid_pred_landmark, valid_gt_landmark)

class NLL_OHEM(torch.nn.NLLLoss):
    """online hard sample mining"""
    def __init__(self, ratio):
        super(NLL_OHEM).__init__(None, True)
        self.ratio = ratio

    def forward(self, x, y, ratio=None):
        if ratio is not None:
            self.ratio = ratio

        num_inst = x.size(0)
        num_hns = int(self.ratio*num_inst)

        x_ = x.clone()
        inst_losses = torch.autograd.Variable(torch.zeros(num_inst)).cuda()
        
        for idx, label in enumerate(y.data):
            inst_loss[idx] = -x_.data[idx, label]
        _, idxs = inst_losses.topk(num__hns)
        x_hn = x.index_select(0, idxs)
        y_hn = y.index_select(0, idxs)

        return torch.nn.functional.nll_loss(x_hn, y_hn)


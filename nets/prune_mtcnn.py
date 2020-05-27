import os, sys
sys.path.append('.')
from collections import OrderedDict
from operator import itemgetter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

import nets.prunable_layers as pnn

class PNet(nn.Module):
    '''12*12 stride 2'''
    def __init__(self, is_train=False):
        super(PNet, self).__init__()
        self.is_train = is_train

        '''
        conv1: (H-2)*(W-2)*10
        prelu1: (H-2)*(W-2)*10
        
        pool1: ((H-2)/2)*((W-2)/2)*10
        
        conv2: ((H-2)/2-2)*((W-2)/2-2)*16
        prelu2: ((H-2)/2-2)*((W-2)/2-2)*16
        
        conv3: ((H-2)/2-4)*((W-2)/2-4)*32
        prelu3: ((H-2)/2-4)*((W-2)/2-4)*32
        
        conv4_1: ((H-2)/2-4)*((W-2)/2-4)*2
        conv4_2: ((H-2)/2-4)*((W-2)/2-4)*4

        The last feature map size is: (H - 10)/2 = (H - 12)/2 + 1.
        Thus the effect of PNet equals to moving 12*12 convolution window with 
        kernel size 3, stirde 2.
        '''

        self.features = nn.Sequential(OrderedDict([
            ('conv1', pnn.PConv2d(3, 10, 3, 1)),
            ('prelu1', pnn.PPReLU(10)),
            ('pool1', nn.MaxPool2d(2, 2, ceil_mode=False)),
            
            ('conv2', pnn.PConv2d(10, 16, 3, 1)),
            ('prelu2', pnn.PPReLU(16)),

            ('conv3', pnn.PConv2d(16, 32, 3, 1)),
            ('prelu3', pnn.PPReLU(32)),
        ]))

        self.conv4_1 = pnn.PConv2d(32, 2, 1, 1)
        self.conv4_2 = pnn.PConv2d(32, 4, 1, 1)

    def forward(self, x):
        x = self.features(x)
        scores = self.conv4_1(x)
        offsets = self.conv4_2(x)
        
        # append softmax for inference
        if not self.is_train:
            socres = F.softmax(scores, dim=1)

        return scores, offsets
    
    def prune(self, device):
        features = list(self.features)
        per_layer_taylor_estimates = [(module.taylor_estimates, layer_idx) 
            for layer_idx, module in enumerate(features)
            if issubclass(type(module), pnn.PConv2d) and module.out_channels > 1]
        per_filter_taylor_estimates = [(per_filter_estimate, filter_idx, layer_idx)
            for per_layer_estimate, layer_idx in per_layer_taylor_estimates
            for filter_idx, per_filter_estimate in enumerate(per_layer_estimate)]
        
        _, min_filter_idx, min_layer_idx = min(per_filter_taylor_estimates, key=itemgetter(0))
        pconv2d = self.features[min_layer_idx]
        pconv2d.prune_feature_map(min_filter_idx, device)

        prelu = self.features[min_layer_idx+1]
        prelu.drop_input_channel(min_filter_idx, device)
        
        next_conv = None
        next_conv_layer_idx = min_layer_idx+1
        
        while next_conv_layer_idx < len(self.features._modules.items()):
            res = list(self.features._modules.items())[next_conv_layer_idx]
            if isinstance(res[1], pnn.PConv2d):
                next_name, next_conv = res
                break
            next_conv_layer_idx += 1
        if next_conv is not None:
            next_conv.drop_input_channel(min_filter_idx)
    
        # if it's the last conv in self.features
        if min_layer_idx+1 == len(self.features._modules.items())-1:
            self.conv4_1.drop_input_channel(min_filter_idx, device)
            self.conv4_2.drop_input_channel(min_filter_idx, device)

if __name__ == "__main__":
    pnet = PNet(is_train=False)
    summary(pnet.cuda(), (3, 12, 12))

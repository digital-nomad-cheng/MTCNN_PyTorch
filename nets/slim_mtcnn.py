'''PNet, RNet, ONet, inspired from shufflenet and mobilenet'''

import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

def channel_shuffle(x, groups):
    """Shuffle channels, from PyTorch official code

    Parameters: 
    -----------
    x: pytorch tensor have shape N*C*H*C
    groups: num of groups to split channels
    Returns: channel shuffled pytorch tensor
    --------
    """
    batch_size, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    
    # reshape
    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    
    # flatten
    x = x.view(batch_size, -1, height, width)
    
    return x

class ShuffleConv2d(nn.Module):
    def __init__(self, inp, oup, kernel_size=3, stride=2, groups=2):
        super(ShuffleConv2d, self).__init__()
        self.groups = groups
        inp_per_group = inp // groups
        oup_per_group = oup // groups
        self.shuffle_conv = nn.Sequential(
            nn.Conv2d(inp_per_group, inp_per_group, kernel_size=kernel_size,
                stride=stride, groups=inp_per_group),
            nn.PReLU(),
            nn.Conv2d(inp_per_group, oup_per_group, kernel_size=1, stride=1),
            nn.PReLU()
        )

    def forward(self, x):
        x_list = list(x.chunk(self.groups, dim=1))
        for i, x in enumerate(x_list):
            x_list[i] = self.shuffle_conv(x)
        x = torch.cat(x_list, dim=1)
        x = channel_shuffle(x, self.groups)
        return x

class DepthwiseConv2d(nn.Module):
    def __init__(self, inp, oup, stride):
        super(DepthwiseConv2d, self).__init__()
        self.use_res = (inp == oup) and (stride == 1)
        self.feature = nn.Sequential(
            nn.Conv2d(inp, inp, kernel_size=3, stride=stride, groups=inp),
            nn.PReLU(),
            nn.Conv2d(inp, oup, kernel_size=1, stride=1),
            nn.PReLU()
        )
    
    def forward(self, x):
        out = self.feature(x)
        if self.use_res:
            out = out + x
        return out

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        # without this pretrained model won't working
        x = x.transpose(3, 2).contiguous()
        return x.view(x.size(0), -1)

class PNet(nn.Module):
    '''12*12 stride 2'''
    def __init__(self, is_train=False):
        super(PNet, self).__init__()
        
        self.is_train = is_train
        
        self.conv1 = nn.Sequential(
                nn.Conv2d(3, 12, kernel_size=3, stride=1),
                nn.PReLU(),
                nn.MaxPool2d(2, 2, ceil_mode=False)
        )
        self.conv2 = ShuffleConv2d(12, 24, 3, 1, 2)
        self.conv3 = ShuffleConv2d(24, 32, 3, 1, 2)

        self.conv4_1 = nn.Conv2d(32, 2, kernel_size=1)
        self.conv4_2 = nn.Conv2d(32, 4, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        scores = self.conv4_1(x)
        offsets = self.conv4_2(x)
        
        # append softmax for inference
        if not self.is_train:
            scores = F.softmax(scores, dim=1)
        
        return scores, offsets

class RNet(nn.Module):
    def __init__(self, is_train=False):
        super(RNet, self).__init__()
        self.is_train = is_train

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1),
            nn.PReLU(),
            nn.MaxPool2d(3, 2, ceil_mode=False)
        )
        self.conv2 = nn.Sequential(
            ShuffleConv2d(32, 64, 3, 1, 2),
            nn.MaxPool2d(3, 2, ceil_mode=False),
        )
        self.conv3 = nn.Sequential(
            ShuffleConv2d(64, 128, 2, 1, 2)
        )
        self.conv4 = nn.Sequential(
            Flatten(),
            nn.Linear(128*2*2, 128),
            nn.PReLU()
        )
        
        self.conv5_1 = nn.Linear(128, 2)
        self.conv5_2 = nn.Linear(128, 4)
        

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        scores = self.conv5_1(x)
        offsets = self.conv5_2(x)
        
        if not self.is_train:
            scores = F.softmax(scores, dim=1)
        
        return scores, offsets

class ONet(nn.Module):
    def __init__(self, is_train=False, train_landmarks=False):
        super(ONet, self).__init__()
        self.is_train = is_train
        self.train_landmarks = train_landmarks
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            nn.PReLU(32),
            nn.MaxPool2d(3, 2, ceil_mode=False)
        )
        self.conv2 = nn.Sequential(
            ShuffleConv2d(32, 64, 3, 1, 2),
            nn.MaxPool2d(3, 2, ceil_mode=False)
        )
        self.conv3 = nn.Sequential(
            ShuffleConv2d(64, 128, 3, 1, 2),
            nn.MaxPool2d(2, 2, ceil_mode=False)
        )
        self.conv4 = nn.Sequential(
            ShuffleConv2d(128, 256, 3, 1, 2)
        )
        self.conv5 = nn.Sequential(
            Flatten(),
            nn.Linear(256, 256),
            nn.PReLU(256)
        )

        self.conv6_1 = nn.Linear(256, 2)
        self.conv6_2 = nn.Linear(256, 4)
        self.conv6_3 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        
        scores = self.conv6_1(x)
        offsets = self.conv6_2(x)

        if not self.is_train:
            scores = F.softmax(scores, dim=1)
        
        if self.train_landmarks:
            landmarks = self.conv6_3(x)
            
            return scores, offsets, landmarks
        
        return scores, offsets

if __name__ == "__main__":
    #device = torch.device("cuda:1")
    #pnet = PNet(is_train=False)
    #pnet.load_state_dict(torch.load('./pretrained_weights/best_pnet.pth'))
    #torch.onnx.export(pnet, torch.randn(1, 3, 12, 12), './onnx2ncnn/pnet.onnx', 
    #        input_names=['input'], output_names=['scores', 'offsets'])
    #pnet.to(device)
    #summary(pnet, (3, 12, 12))
    
     
    #rnet = RNet(is_train=False)
    #rnet.load_state_dict(torch.load('./pretrained_weights/best_rnet.pth'))
    #torch.onnx.export(rnet, torch.randn(1, 3, 24, 24), './onnx2ncnn/rnet.onnx', 
    #        input_names=['input'], output_names=['scores', 'offsets'])
    #summary(rnet.cuda(), (3, 24, 24)) 
    
    onet = ONet(is_train=False)
    #onet.load_state_dict(torch.load('./pretrained_weights/best_onet.pth'))
    #summary(onet.cuda(), (3, 48, 48))
    torch.onnx.export(onet, torch.randn(1, 3, 48, 48), './onnx2ncnn/onet.onnx', 
            input_names=['input'], output_names=['scores', 'offsets'])

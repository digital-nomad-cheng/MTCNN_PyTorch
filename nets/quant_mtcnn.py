from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import brevitas.nn as qnn
from brevitas.core.quant import QuantType
from torchsummary import summary

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        # without this pretrained model won't working
        # x = x.transpose(3, 2).contiguous()
        return x.view(x.size(0), -1)

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
            ('conv1', qnn.QuantConv2d(3, 10, 3, 1, 
                weight_quant_type=QuantType.INT, weight_bit_width=8)),
            ('prelu1', qnn.QuantReLU(quant_type=QuantType.INT, bit_width=8, max_val=6)),
            ('pool1', nn.MaxPool2d(2, 2, ceil_mode=False)),
            
            ('conv2', qnn.QuantConv2d(10, 16, 3, 1,
                weight_quant_type=QuantType.INT, weight_bit_width=8)),
            ('prelu2', qnn.QuantReLU(quant_type=QuantType.INT, bit_width=8, max_val=6)),

            ('conv3', qnn.QuantConv2d(16, 32, 3, 1,
                weight_quant_type=QuantType.INT, weight_bit_width=8)),
            ('prelu3', qnn.QuantReLU(quant_type=QuantType.INT, bit_width=8, max_val=6)),
        ]))

        self.conv4_1 = qnn.QuantConv2d(32, 2, 1, 1, 
                weight_quant_type=QuantType.INT, weight_bit_width=8)
        self.conv4_2 = qnn.QuantConv2d(32, 4, 1, 1,
                weight_quant_type=QuantType.INT, weight_bit_width=8)

    def forward(self, x):
        x = self.features(x)
        scores = self.conv4_1(x)
        offsets = self.conv4_2(x)
        
        # append softmax for inference
        if not self.is_train:
            socres = F.softmax(scores, dim=1)
        
        return scores, offsets

class RNet(nn.Module):
    '''Input size should be 24*24*3'''
    def __init__(self, is_train=False):
        super(RNet, self).__init__()
        self.is_train = is_train
        
        self.features = nn.Sequential(OrderedDict([
            ('conv1', qnn.QuantConv2d(3, 28, 3, 1,
                weight_quant_type=QuantType.INT, weight_bit_width=8)), # 24 -2 = 22
            ('prelu1', qnn.QuantReLU(quant_type=QuantType.INT, bit_width=8, max_val=6)),
            ('pool1', nn.MaxPool2d(3, 2, ceil_mode=False)), # (22-3)/2 + 1 = 10

            ('conv2', qnn.QuantConv2d(28, 48, 3, 1,
                weight_quant_type=QuantType.INT, weight_bit_width=8)), # 10 - 2 = 8
            ('prelu2', qnn.QuantReLU(quant_type=QuantType.INT, bit_width=8, max_val=6)),
            ('pool2', nn.MaxPool2d(3, 2, ceil_mode=False)), # (8-3)/2 + 1 = 3

            ('conv3', qnn.QuantConv2d(48, 64, 2, 1,
                weight_quant_type=QuantType.INT, weight_bit_width=8)), #  3 - 1 = 2
            ('prelu3', qnn.QuantReLU(quant_type=QuantType.INT, bit_width=8, max_val=6)),

            ('flatten', Flatten()),
            ('conv4', qnn.QuantLinear(64*2*2, 128, 
                weight_quant_type=QuantType.INT, bias=False, weight_bit_width=8)),
            ('prelu4', qnn.QuantReLU(quant_type=QuantType.INT, bit_width=8, max_val=6)),
            #('dropout', nn.Dropout(0.2))
        ]))

        self.conv5_1 = qnn.QuantLinear(128, 2,
                weight_quant_type=QuantType.INT, bias=False, weight_bit_width=8)
        self.conv5_2 = qnn.QuantLinear(128, 4,
                weight_quant_type=QuantType.INT, bias=False, weight_bit_width=8)

    def forward(self, x):
        x = self.features(x)
        scores = self.conv5_1(x)
        offsets = self.conv5_2(x)

        if not self.is_train:
            scores = F.softmax(scores, dim=1)
        return scores, offsets

class ONet(nn.Module):
    '''Input size should be 48*48*3'''
    def __init__(self, is_train=False, train_landmarks=False):
        super(ONet, self).__init__()
        
        self.is_train = is_train
        self.train_landmarks = train_landmarks

        self.features = nn.Sequential(OrderedDict([
            ('conv1', qnn.QuantConv2d(3, 32, 3, 1,
                weight_quant_type=QuantType.INT, weight_bit_width=8)), # 48 - 2 = 46
            ('prelu1', qnn.QuantReLU(quant_type=QuantType.INT, bit_width=8, max_val=6)),
            ('pool1', nn.MaxPool2d(3, 2, ceil_mode=False)), # (46-3)/2 + 1 = 22 

            ('conv2', qnn.QuantConv2d(32, 64, 3, 1,
                weight_quant_type=QuantType.INT, weight_bit_width=8)), # 22 - 2 = 20
            ('prelu2', qnn.QuantReLU(quant_type=QuantType.INT, bit_width=8, max_val=6)),
            ('pool2', nn.MaxPool2d(3, 2, ceil_mode=False)), # (20-3)/2 + 1 = 9

            ('conv3', qnn.QuantConv2d(64, 64, 3, 1,
                weight_quant_type=QuantType.INT, bit_width=8, max_val=6)), # 9 - 2 = 7   
            ('prelu3', qnn.QuantReLU(quant_type=QuantType.INT, bit_width=8, max_val=6)),
            ('pool3', nn.MaxPool2d(2, 2, ceil_mode=False)), # (7-2)/2 + 1 = 3
            
            ('conv4', qnn.QuantConv2d(64, 128, 2, 1,
                weight_quant_type=QuantType.INT, bit_width=8, max_val=6)), # 3 - 1 = 2
            ('prelu4', qnn.QuantReLU(quant_type=QuantType.INT, bit_width=8, max_val=6)),

            ('flatten', Flatten()),
            ('conv5', qnn.QuantLinear(128*2*2, 256,
                weight_quant_type=QuantType.INT, bias=False, weight_bit_width=8)),
            ('prelu5', qnn.QuantReLU(quant_type=QuantType.INT, bit_width=8, max_val=6)),
            ('dropout', nn.Dropout(0.2))
        ]))

        self.conv6_1 = qnn.QuantLinear(256, 2,
                weight_quant_type=QuantType.INT, bias=False, weight_bit_width=8)
        self.conv6_2 = qnn.QuantLinear(256, 4, 
                weight_quant_type=QuantType.INT, bias=False, weight_bit_width=8)
        self.conv6_3 = qnn.QuantLinear(256, 10,
                weight_quant_type=QuantType.INT, bias=False, weight_bit_width=8)
    
    def forward(self, x):
        x = self.features(x)
        scores = self.conv6_1(x)
        offsets = self.conv6_2(x)
        
        if not self.is_train:
            scores = F.softmax(scores, dim=1)
        
        if self.train_landmarks:
            landmarks = self.conv6_3(x)
            return scores, offsets, landmarks
       
        return scores, offsets


if __name__ == "__main__":
    pnet = PNet(is_train=False)
    print(pnet)
    pnet(torch.randn(1, 3, 12, 12))
    #pnet.load_state_dict(torch.load('./pretrained_weights/mtcnn/best_pnet.pth'))
    #torch.onnx.export(pnet, torch.randn(1, 3, 12, 12), './onnx2ncnn/pnet.onnx', 
    #        input_names=['input'], output_names=['scores', 'offsets'])
    #summary(pnet.cuda(), (3, 12, 12))
    
    rnet = RNet(is_train=False)
    print(rnet)
    rnet(torch.randn(1, 3, 24, 24))
    #rnet.load_state_dict(torch.load('./pretrained_weights/mtcnn/best_rnet.pth'))
    #torch.onnx.export(rnet, torch.randn(1, 3, 24, 24), './onnx2ncnn/rnet.onnx', 
    #        input_names=['input'], output_names=['scores', 'offsets'])
    #summary(rnet.cuda(), (3, 24, 24)) 
    
    #onet = ONet(is_train=False)
    #onet.load_state_dict(torch.load('./pretrained_weights/mtcnn/best_onet.pth'))
    #torch.onnx.export(onet, torch.randn(1, 3, 48, 48), './onnx2ncnn/onet.onnx',
    #        input_names=['input'], output_names=['scores', 'offsets'])
    #summary(onet.cuda(), (3, 48, 48))

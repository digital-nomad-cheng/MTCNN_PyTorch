import os
import time

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

from nets.mtcnn import ONet
from nets.mtcnn import PNet
from nets.mtcnn import RNet
import tools.utils as utils
import config

class MTCNNDetector(object):
    ''' P, R, O net for face detection and alignment'''

    def __init__(self,
                 p_model_path=None,
                 r_model_path=None,
                 o_model_path=None,
                 min_face_size=12,
                 stride=2,
                 threshold=[0.6, 0.7, 0.7],
                 scale_factor=0.709,
                 use_cuda=True):

        self.pnet_detector, self.rnet_detector, self.onet_detector = self.create_mtcnn_net(
            p_model_path, r_model_path, o_model_path, use_cuda)
        
        self.min_face_size = min_face_size
        self.stride = stride
        self.thresh = threshold
        self.scale_factor = scale_factor

    def create_mtcnn_net(self, p_model_path=None, r_model_path=None, o_model_path=None, use_cuda=True):
        '''Create MTCNN Pnet, Rnet, Onet, load weights if there are any.'''
        pnet, rnet, onet = None, None, None
        self.device = torch.device(
            "cuda:1" if use_cuda and torch.cuda.is_available() else "cpu")

        if p_model_path is not None:
            pnet = PNet()
            pnet.load_state_dict(torch.load(p_model_path))
            if (use_cuda):
                pnet.to(self.device)
            pnet.eval()

        if r_model_path is not None:
            rnet = RNet()
            rnet.load_state_dict(torch.load(r_model_path))
            if (use_cuda):
                rnet.to(self.device)
            rnet.eval()

        if o_model_path is not None:
            onet = ONet()
            onet.load_state_dict(torch.load(o_model_path))
            if (use_cuda):
                onet.to(self.device)
            onet.eval()

        return pnet, rnet, onet

    def generate_bounding_box(self, cls_map, bbox_map, scale, threshold):
        '''
        generate bounding bboxes from feature map
        for PNet, there exists no fc layer, only convolution layer, 
            so feature map n x m x 2/4, 2 for classification, 4 for bboxes 
        
        Parameters:
        -----------
            cls_map: numpy array , 1 x n x m x 2, detect score for each position
            bbox_map: numpy array , 1 x n x m x 4, detect bbox regression value for each position
            scale: float number, scale of this detection
            threshold: float number, detect threshold
        Returns:
        --------
            bbox array
        '''
        stride = config.STRIDE
        cellsize = config.PNET_SIZE
        # softmax layer 1 for face, return a tuple with an array of row idxs and
        # an array of col idxs
        # locate face above threshold from cls_map
        t_index = np.where(cls_map[0, :, :, 1] > threshold)
        
        # find nothing
        if t_index[0].size == 0:
            return np.array([])

        dx1, dy1, dx2, dy2 = [bbox_map[0, t_index[0], t_index[1], i]
                              for i in range(4)]
        bbox_map = np.array([dx1, dy1, dx2, dy2])

        score = cls_map[0, t_index[0], t_index[1], 1]
        boundingbox = np.vstack([np.round((stride * t_index[1] + 1.0) / scale),
                                 np.round((stride * t_index[0] + 1.0) / scale),
                                 np.round(
                                     (stride * t_index[1] + cellsize) / scale),
                                 np.round(
                                     (stride * t_index[0] + cellsize) / scale),
                                 score,
                                 bbox_map,
                                 ])

        return boundingbox.T

    def resize_image(self, img, scale):
        """
        Resize image and transform dimention to [batchsize, channel, height, width]
        Parameters:
        ----------
            img: numpy array , height x width x channel, input image, channels in BGR order here
            scale: float number, scale factor of resize operation
        Returns:
        -------
            transformed image tensor , 1 x channel x height x width
        """
        height, width, channels = img.shape
        new_height = int(height * scale)  # resized new height
        new_width = int(width * scale)  # resized new width
        new_size = (new_width, new_height)
        img_resized = cv2.resize(
            img, new_size, interpolation=cv2.INTER_LINEAR)  # resized image
        return img_resized

    def detect_pnet(self, im):
        """Get face candidates through pnet

        Parameters:
        -----------
        im: numpy array, input image array

        Returns:
        --------
        bboxes_align: numpy array
            bboxes after calibration
        """
        h, w, c = im.shape
        net_size = config.PNET_SIZE
        current_scale = float(net_size) / self.min_face_size  # find initial scale
        im_resized = self.resize_image(im, current_scale)
        current_height, current_width, _ = im_resized.shape

        # bounding boxes for all the pyramid scales
        all_bboxes = list()
        # generating bounding boxes for each scale
        while min(current_height, current_width) > net_size:
            image_tensor = utils.convert_image_to_tensor(im_resized)
            feed_imgs = image_tensor.unsqueeze(0)
            feed_imgs = feed_imgs.to(self.device)

            cls_map, reg_map = self.pnet_detector(feed_imgs)
            cls_map_np = utils.convert_chwTensor_to_hwcNumpy(cls_map.cpu())
            reg_map_np = utils.convert_chwTensor_to_hwcNumpy(reg_map.cpu())
            
            bboxes = self.generate_bounding_box(
                cls_map_np, reg_map_np, current_scale, self.thresh[0])

            current_scale *= self.scale_factor
            im_resized = self.resize_image(im, current_scale)
            current_height, current_width, _ = im_resized.shape

            if bboxes.size == 0:
                continue

            keep = utils.nms(bboxes[:, :5], 0.5, 'Union')
            bboxes = bboxes[keep]
            all_bboxes.append(bboxes)

        
        if len(all_bboxes) == 0:
            return None

        all_bboxes = np.vstack(all_bboxes)

        # apply nms to the detections from all the scales 
        keep = utils.nms(all_bboxes[:, 0:5], 0.5, 'Union')
        all_bboxes = all_bboxes[keep]
        
        # 0-4: original bboxes, 5: score, 5: offsets
        bboxes_align = utils.calibrate_box(all_bboxes[:, 0:5], all_bboxes[:, 5:])
        bboxes_align = utils.convert_to_square(bboxes_align)  
        bboxes_align[:, 0:4] = np.round(bboxes_align[:, 0:4])
        
        return bboxes_align

    def detect_rnet(self, im, bboxes):
        """Get face candidates using rnet

        Parameters:
        ----------
        im: numpy array
            input image array
        bboxes: numpy array
            detection results of pnet

        Returns:
        -------
        bboxes_align: numpy array
            bboxes after calibration
        """
        net_size = config.RNET_SIZE
        h, w, c = im.shape
        if bboxes is None:
            return None

        num_bboxes = bboxes.shape[0]
        
        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = utils.correct_bboxes(bboxes, w, h)

        
        # crop face using pnet proposals
        cropped_ims_tensors = []
        for i in range(num_bboxes):
            try:
                if tmph[i] > 0 and tmpw[i] > 0:
                    tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
                    tmp[dy[i]:edy[i], dx[i]:edx[i], :] = im[y[i]:ey[i], x[i]:ex[i], :]
                    crop_im = cv2.resize(tmp, (net_size, net_size))
                    crop_im_tensor = utils.convert_image_to_tensor(crop_im)
                    cropped_ims_tensors.append(crop_im_tensor)
            except ValueError as e:
                print('dy: {}, edy: {}, dx: {}, edx: {}'.format(dy[i], edy[i], dx[i], edx[i]))
                print('y: {}, ey: {}, x: {}, ex: {}'.format(y[i], ey[i], x[i], ex[i]))
                print(e)

        # provide input tensor, if there are too many proposals in PNet
        # there might be OOM
        feed_imgs = torch.stack(cropped_ims_tensors)
        feed_imgs = feed_imgs.to(self.device)
        print("feed_imgs shape:", feed_imgs.shape) 
        cls, reg = self.rnet_detector(feed_imgs)
        cls = cls.cpu().data.numpy()
        reg = reg.cpu().data.numpy()
        
        # for i in range(cls.shape[0]):
        #    print(cls[i])
        
        keep_inds = np.where(cls[:, 1] > self.thresh[1])[0]
        if len(keep_inds) > 0:
            keep_bboxes = bboxes[keep_inds]
            keep_cls = cls[keep_inds, :]
            keep_reg = reg[keep_inds]
            # using softmax 1 as cls score from Rnet
            keep_bboxes[:, 4] = keep_cls[:, 1].reshape((-1,))
        else:
            return None
        print("rnet threshold shape:", keep_bboxes.shape[0])

        keep = utils.nms(keep_bboxes, 0.7, "Minimum")
        if len(keep) == 0:
            return None
       
        keep_cls = keep_cls[keep]
        keep_bboxes = keep_bboxes[keep]
        keep_reg = keep_reg[keep]
        
        print("rnet nms shape:", keep_bboxes.shape[0])

        bboxes_align = utils.calibrate_box(keep_bboxes, keep_reg)
        bboxes_align = utils.convert_to_square(bboxes_align)
        bboxes_align[:, 0:4] = np.round(bboxes_align[:, 0:4]) 

        return bboxes_align

    def detect_onet(self, im, bboxes):
        """Get face candidates using onet

        Parameters:
        ----------
        im: numpy array
            input image array
        bboxes: numpy array
            detection results of rnet

        Returns:
        -------
        bboxes_align: numpy array
            bboxes after calibration
        """
        net_size = config.ONET_SIZE
        h, w, c = im.shape
        if bboxes is None:
            return None

        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = utils.correct_bboxes(bboxes, w, h)
        num_bboxes = bboxes.shape[0]
        
        # crop face using rnet proposal
        cropped_ims_tensors = []
        for i in range(num_bboxes):
            try:
                if tmph[i] > 0 and tmpw[i] > 0:
                    tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
                    tmp[dy[i]:edy[i], dx[i]:edx[i], :] = im[y[i]:ey[i], x[i]:ex[i], :]
                    crop_im = cv2.resize(tmp, (net_size, net_size))
                    cv2.imshow('onet_crop_im_'+str(i), crop_im)
                    crop_im_tensor = utils.convert_image_to_tensor(crop_im)
                    cropped_ims_tensors.append(crop_im_tensor)
            except ValueError as e:
                print(e)
        cv2.waitKey(0)

        feed_imgs = torch.stack(cropped_ims_tensors)
        feed_imgs = feed_imgs.to(self.device)
        print("feed_imgs shape:", feed_imgs.shape[0])
        cls, reg = self.onet_detector(feed_imgs)
        cls = cls.cpu().data.numpy()
        reg = reg.cpu().data.numpy()

        keep_inds = np.where(cls[:, 1] > self.thresh[2])[0]
        if len(keep_inds) > 0:
            keep_bboxes = bboxes[keep_inds]
            keep_cls = cls[keep_inds, :]
            keep_reg = reg[keep_inds]
            keep_bboxes[:, 4] = keep_cls[:, 1].reshape((-1,))
        else:
            return None
        print("Onet threshold shape:", keep_bboxes.shape[0]) 
        
        bboxes_align = utils.calibrate_box(keep_bboxes, keep_reg)
        keep = utils.nms(bboxes_align, 0.7, mode='Minimum')
        
        if len(keep) == 0:
            return None

        bboxes_align = bboxes_align[keep]
        bboxes_align = utils.convert_to_square(bboxes_align)
        return bboxes_align

    def detect_face(self, img):
        ''' Detect face over image '''
        bboxes_align = np.array([])

        t = time.time()

        # pnet
        if self.pnet_detector:
            bboxes_align = self.detect_pnet(img)
            if bboxes_align is None:
                print("No faces in this image according to pnet")
                return np.array([])
            t1 = time.time() - t
            t = time.time()
        print("Number of faces from pnet:", bboxes_align.shape[0])
        # rnet
        if self.rnet_detector:
            bboxes_align = self.detect_rnet(img, bboxes_align)
            if bboxes_align is None:
                print("No faces in this image according to rnet")
                return np.array([])
            t2 = time.time() - t
            t = time.time()
        print("Number of faces from rnet:", bboxes_align.shape[0])
        # onet
        if self.onet_detector:
            bboxes_align = self.detect_onet(img, bboxes_align)
            if bboxes_align is None:
                print("No faces in this image according to onet")
                return np.array([])
            t3 = time.time() - t
            t = time.time()
        
            print("time cost " + '{:.3f}'.format(t1 + t2 + t3) + \
                '  pnet {:.3f}  rnet {:.3f}  onet {:.3f}'.format(t1, t2, t3))

        print("Number of faces from onet:", bboxes_align.shape[0])
        
        return bboxes_align

import time
import os, sys
sys.path.append('.')
import pathlib
import logging

import cv2
import torch 

from MTCNN import MTCNNDetector
import config

logger = logging.getLogger("app")
formatter = logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s')
console_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(console_handler)
logger.setLevel(logging.INFO)
console_handler.formatter = formatter  # 也可以直接给formatter赋值 


def draw_images(img, bboxs):  # 在图片上绘制人脸框及特征点
    num_face = bboxs.shape[0]
    for i in range(num_face):
        cv2.rectangle(img, (int(bboxs[i, 0]), int(bboxs[i, 1])), (int(
            bboxs[i, 2]), int(bboxs[i, 3])), (0, 255, 0), 3)
    return img


if __name__ == '__main__':
    base_model_path = './pretrained_weights/mtcnn'
    mtcnn_detector = MTCNNDetector(
        p_model_path=os.path.join(base_model_path, 'best_pnet.pth'), 
        r_model_path=os.path.join(base_model_path, 'best_rnet.pth'),
        o_model_path=os.path.join(base_model_path, 'best_onet.pth'),
        min_face_size=24, threshold=[0.7, 0.8, 0.9], use_cuda=False) 
    logger.info("Init the MtcnnDetector.")
    
    cap = cv2.VideoCapture('./test/test_video.mov')
    if not cap.isOpened():
        print("Failed to open capture from file")
    start = time.time()
    num = 0
    while(cap.isOpened):
        ret, img = cap.read()
        logger.info("Start to process No.{} image.".format(num))
        RGB_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #RGB_image = cv2.medianBlur(RGB_image, 5)
        bboxs = mtcnn_detector.detect_face(RGB_image)
        img = draw_images(img, bboxs)
        cv2.imshow('frame', img)
        cv2.waitKey(1)
        num += 1
    logger.info("Finish all the images.")
    logger.info("Elapsed time: {:.3f}s".format(time.time() - start))

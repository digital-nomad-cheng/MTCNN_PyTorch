import os, sys
sys.path.append('.')

import cv2
import numpy as np

from tools.utils import *
from MTCNN import MTCNNDetector
#from MTCNN_debug import MTCNNDetector

if __name__ == "__main__":
    base_model_path = './pretrained_weights/mtcnn'
    mtcnn_detector = MTCNNDetector(
        p_model_path=os.path.join(base_model_path, 'best_pnet.pth'),
        r_model_path=os.path.join(base_model_path, 'best_rnet.pth'),
        o_model_path=os.path.join(base_model_path, 'best_onet_landmark.pth'),
        min_face_size=40,
        threshold=[0.7, 0.8, 0.9]
    )

    #image_path = './data/user/images'
    image_path='./test'
    images = [f for f in os.listdir(image_path) if f.endswith('.jpg')]
    
    for image in images:
        image = cv2.imread(os.path.join(image_path, image), 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bboxes, landmarks = mtcnn_detector.detect_face(image)
        for i in range(bboxes.shape[0]):
            x0, y0, x1, y1 = bboxes[i, :4]
            width = int(x1 - x0 + 1)
            height = int(y1 - y0 + 1)
            cv2.rectangle(image, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 255), 1)
            for j in range(5):
                x, y = int(x0 + landmarks[i, j]*width), int(y0 + landmarks[i, j+5]*height)
                cv2.circle(image, (x, y), 2, (255, 0, 255), 2)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow('image', image)
        cv2.waitKey(0)

import os, sys
sys.path.append('.')

import cv2
import numpy as np

from tools.utils import *
from MTCNN import MTCNNDetector

if __name__ == "__main__":
    base_model_path = './pretrained_weights/mtcnn'
    mtcnn_detector = MTCNNDetector(
        p_model_path=os.path.join(base_model_path, 'best_pnet.pth'),
        r_model_path=os.path.join(base_model_path, 'best_rnet.pth'),
        o_model_path=os.path.join(base_model_path, 'best_onet.pth'),
        min_face_size=40,
        threshold=[0.99, 0.9, 0.8]
    )

    image_path = './data/user/images'
    images = [f for f in os.listdir(image_path) if f.endswith('.jpg')]
    
    for image in images:
        image = cv2.imread(os.path.join(image_path, image), 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bboxes = mtcnn_detector.detect_face(image)
        for i in range(bboxes.shape[0]):
            x0, y0, x1, y1 = bboxes[i, :4]
            cv2.rectangle(image, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 255), 1)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow('image', image)
        cv2.waitKey(0)

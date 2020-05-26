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
        threshold=[0.7, 0.8, 0.9]
    )
    fddb_path = './data/FDDB'
    for i in range(1, 11):
        with open(os.path.join(fddb_path, 'FDDB-folds/imgpath', 'FDDB-fold-{:02d}.txt'.format(i)), 'r') as f:
            lines = f.readlines()
            for line in lines:
                image_path = line.strip() + '.jpg'
                print(image_path)
                image = cv2.imread(os.path.join(fddb_path, image_path))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                bboxes = mtcnn_detector.detect_face(image)
                for i in range(bboxes.shape[0]):
                    x0, y0, x1, y1 = bboxes[i, :4]
                    cv2.rectangle(image, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 255), 1)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.imshow('image', image)
                cv2.waitKey(0)

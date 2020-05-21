import os

from easydict import EasyDict as edict
'''
MODEL_STORE_DIR = "./models"
ANNO_STORE_DIR = "./annotations"
TRAIN_DATA_DIR = "/dev/data/MTCNN"
'''

'''
# -------- generate training dataset ---------- #
PROB_THRESH = 0.15 # for ./annotations/wider_prob.txt' threshold

USE_CELEBA = False
USE_USER = False

SAVE_PREFIX = 'wider'

WIDER_DATA_PATH = '/home/idealabs/data/opensource_dataset/WIDER/WIDER_train/images'
WIDER_DATA_ANNO_FILE = 'wider_anno_tmp.txt'
WIDER_DATA_PROB_FILE = 'wider_prob.txt'
WIDER_VAL_DATA_ANNO_FILE = 'wider_anno_val.txt'

USER_DATA_PATH = '/home/idealabs/data/opensource_dataset/user/imgs'
USER_DATA_ANNO_FILE = 'user_anno.txt' 

CELEBA_DATA_PATH = '/home/idealabs/data/opensource_dataset/celeba/img_celeba'
CELEBA_DATA_ANNO_FILE = 'small_celeba_anno.txt' 
'''

# --------------- training MTCNN config ------------- #
ANNO_PATH = './annotations'

STRIDE = 2
PNET_SIZE = 12
PNET_POSTIVE_ANNO_FILENAME = "pos_{0}.txt".format(PNET_SIZE)
PNET_NEGATIVE_ANNO_FILENAME = "neg_{0}.txt".format(PNET_SIZE)
PNET_PART_ANNO_FILENAME = "part_{0}.txt".format(PNET_SIZE)
PNET_LANDMARK_ANNO_FILENAME = "landmark_{0}.txt".format(PNET_SIZE)

RNET_SIZE = 24
RNET_POSTIVE_ANNO_FILENAME = "pos_{0}.txt".format(RNET_SIZE)
RNET_NEGATIVE_ANNO_FILENAME = "neg_{0}.txt".format(RNET_SIZE)
RNET_PART_ANNO_FILENAME = "part_{0}.txt".format(RNET_SIZE)
RNET_LANDMARK_ANNO_FILENAME = "landmark_{0}.txt".format(RNET_SIZE)

ONET_SIZE = 48
ONET_POSTIVE_ANNO_FILENAME = "pos_{0}.txt".format(ONET_SIZE)
ONET_NEGATIVE_ANNO_FILENAME = "neg_{0}.txt".format(ONET_SIZE)
ONET_PART_ANNO_FILENAME = "part_{0}.txt".format(ONET_SIZE)
ONET_LANDMARK_ANNO_FILENAME = "landmark_{0}.txt".format(ONET_SIZE)

PNET_TRAIN_IMGLIST_FILENAME = "imglist_anno_{0}_train.txt".format(PNET_SIZE)
RNET_TRAIN_IMGLIST_FILENAME = "imglist_anno_{0}_train.txt".format(RNET_SIZE)
ONET_TRAIN_IMGLIST_FILENAME = "imglist_anno_{0}_train.txt".format(ONET_SIZE)
PNET_VAL_IMGLIST_FILENAME = 'imglist_anno_{0}_val.txt'.format(PNET_SIZE)
RNET_VAL_IMGLIST_FILENAME = 'imglist_anno_{0}_val.txt'.format(RNET_SIZE)
ONET_VAL_IMGLIST_FILENAME = 'imglist_anno_{0}_val.txt'.format(ONET_SIZE)


USE_CUDA = True
BATCH_SIZE = 1024
LR = 0.01
EPOCHS = 100
STEPS = [10, 40, 80]

# --------------------- tracking -----------------------# 
'''
TRACE = edict()                                                            
TRACE.ema_or_one_euro='euro'                 ### post process             
TRACE.pixel_thres=1                                                       
TRACE.smooth_box=0.3                         ## if use euro, this will be disable
TRACE.smooth_landmark=0.95                   ## if use euro, this will be disable
TRACE.iou_thres=0.5                                                       
'''                           

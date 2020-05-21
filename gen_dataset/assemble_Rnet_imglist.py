import os
import sys
sys.path.append(os.getcwd())
import assemble

mode = 'val'


rnet_postive_file = 'annotations/pos_24_{}.txt'.format(mode)
rnet_part_file = 'annotations/part_24_{}.txt'.format(mode)
rnet_neg_file = 'annotations/neg_24_{}.txt'.format(mode)
# pnet_landmark_file = './annotations/landmark_12.txt'
imglist_filename = 'annotations/imglist_anno_24_{}.txt'.format(mode)

if __name__ == '__main__':

    anno_list = []

    anno_list.append(rnet_postive_file)
    anno_list.append(rnet_part_file)
    anno_list.append(rnet_neg_file)
    # anno_list.append(pnet_landmark_file)

    chose_count = assemble.assemble_data(imglist_filename ,anno_list)
    print("RNet train annotation result file path:%s" % imglist_filename)

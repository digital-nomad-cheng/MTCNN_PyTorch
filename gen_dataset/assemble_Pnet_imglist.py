import os
import sys
sys.path.append(os.getcwd())
import assemble

mode = 'train'

pnet_postive_file = 'annotations/pos_12_{}.txt'.format(mode)
pnet_part_file = 'annotations/part_12_{}.txt'.format(mode)
pnet_neg_file = 'annotations/neg_12_{}.txt'.format(mode)
# pnet_landmark_file = './anno_store/landmark_12.txt'
imglist_filename = 'annotations/imglist_anno_12_{}.txt'.format(mode)

if __name__ == '__main__':

    anno_list = []

    anno_list.append(pnet_postive_file)
    anno_list.append(pnet_part_file)
    anno_list.append(pnet_neg_file)
    # anno_list.append(pnet_landmark_file)

    chose_count = assemble.assemble_data(imglist_filename ,anno_list)
    print("PNet train annotation result file path:%s" % imglist_filename)

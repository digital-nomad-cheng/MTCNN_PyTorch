import os
import sys
sys.path.append(os.getcwd())
import assemble
import config

mode = 'val'
net_size = config.PNET_SIZE

pnet_postive_file = 'annotations/pos_{}_{}.txt'.format(net_size, mode)
pnet_part_file = 'annotations/part_{}_{}.txt'.format(net_size, mode)
pnet_neg_file = 'annotations/neg_{}_{}.txt'.format(net_size, mode)
# pnet_landmark_file = './anno_store/landmark_12.txt'
imglist_filename = 'annotations/imglist_anno_{}_{}.txt'.format(net_size, mode)

if __name__ == '__main__':

    anno_list = []

    anno_list.append(pnet_postive_file)
    anno_list.append(pnet_part_file)
    anno_list.append(pnet_neg_file)
    # anno_list.append(pnet_landmark_file)

    chose_count = assemble.assemble_data(imglist_filename ,anno_list)
    print("PNet train annotation result file path:%s" % imglist_filename)

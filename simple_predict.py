import argparse
import os
import shutil
import glob

import random
import json
import numpy as np
import pandas as pd

from PIL import Image
import pickle

from tqdm import tqdm

from pathlib import Path

from mmdet.ops.nms import nms_cpu, nms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("list_all_classes_path", default='all_casses.txt')
    parser.add_argument("list_valid_classes_path", default='valid_classes_stage0.txt')
    parser.add_argument("mmdet_predict_path",
                        default='/group-volume/orc_srr/multimodal/iceblood/develop/mmdetection/cascade_skolkovo_fit_frozen4_lll_lr_38.pkl')
    parser.add_argument("submit_annotation_path",
                        default='/group-volume/orc_srr/multimodal/iceblood/datasets/main/TEST/final_skolkovo.pickle')
    parser.add_argument("confidence", type=float, default=0.55)
    parser.add_argument("class_accuracy", type=int, default=2)
    parser.add_argument("output_result_path", default='submits')

    args = parser.parse_args()

    pred_pickle_path = args.mmdet_predict_path
    subm_annotations_path = args.submit_annotation_path
    output_result_path = args.output_result_path
    class_accuracy = args.class_accuracy
    convert_class = lambda x: '.'.join(str(x).split('.')[:class_accuracy])

    output_result_path = os.path.join(output_result_path, 'subm_' + pred_pickle_path.split('/')[-1]
                                      + '_' + str(args.confidence)
                                      + '_' + str(args.class_accuracy))
    all_classes_list = []
    with open(args.list_all_classes_path, 'r') as fileobj:
        for row in fileobj:
            all_classes_list.append(row.rstrip('\n'))

    valid_classes_list = []
    with open(args.list_valid_classes_path, 'r') as fileobj:
        for row in fileobj:
            valid_classes_list.append(row.rstrip('\n'))

    all_classes_list = [convert_class(x) for x in all_classes_list]
    valid_classes_list = [convert_class(x) for x in valid_classes_list]

    with open(pred_pickle_path, 'rb') as outfile:
        subm_data = pickle.load(outfile)

    with open(subm_annotations_path, 'rb') as outfile:
        subm_annotations = pickle.load(outfile)

    with open(output_result_path, 'w') as f:
        f.write('\t'.join(['frame', 'xtl', 'ytl', 'xbr', 'ybr', 'class']) + '\n')
        for cur_pred, name in zip(subm_data, subm_annotations):
            name = str('/'.join(name['filename'].split('/')[-2:]))
            img_name = name.split('/')
            img_name = img_name[-2] + '/' + img_name[-1]
            img_name = img_name.replace('.jpg', '')

            all_boxes = []
            all_classes = []
            for cur_sign, cur_boxes in zip(all_classes_list, cur_pred):
                for cur_box in cur_boxes:
                    x, y, xmax, ymax, p = cur_box
                    h = xmax - x
                    w = ymax - y
                    cur_box = [x, y, x + h, y + w, p]
                    if cur_box[-1] >= args.confidence and cur_sign in valid_classes_list:
                        all_boxes.append(cur_box)
                        all_classes.append(cur_sign)

            all_boxes = np.array(all_boxes)
            all_classes = np.array(all_classes)

            if not len(all_boxes):
                continue
            filtered_boxes = nms(all_boxes, 0.05)[1]

            for cur_box, cur_sign in zip(all_boxes[filtered_boxes], all_classes[filtered_boxes]):
                cur_box = cur_box[:4].astype(np.int32)
                cur_box = list(map(str, cur_box.tolist()))
                f.write('\t'.join([img_name, *cur_box, cur_sign]) + '\n')

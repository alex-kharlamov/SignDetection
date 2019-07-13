import os
import json
import pandas as pd
from PIL import Image
from tqdm import *
import numpy as np
import pickle
from shutil import copyfile
from mmdet.apis import init_detector, inference_detector, show_result
import glob

import torch.utils.data as data

from PIL import Image
import os
import os.path

import cv2
import sys
from copy import deepcopy
import torchvision
import torch
from mmdet.ops.nms import nms_cpu, nms
from skimage.feature import match_template
from skimage.color import rgb2gray
import numpy as np
import time


def hw_to_min_max(box):
    return list(map(float, [box[0], box[1], box[2]+box[0], box[3]+box[1]]))

def min_max_to_hw(cur_box):
    return (cur_box[0], cur_box[1], cur_box[2] - cur_box[0], cur_box[3] - cur_box[1])

def square_from_hw(cur_box):
    return cur_box[2] * cur_box[3]

def scale_translate(image, x, y):
    # Define the translation matrix and perform the translation
    M = np.float32([[1, 0, x], [0, 1, y]])
    shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    # Return the translated image
    return shifted

def scale_rotate(image, angle, center = None, scale = 1.0):
    # Grab the dimensions of the image
    (h, w) = image.shape[:2]

    # If the center is None, initialize it as the center of
    # the image
    if center is None:
        center = (w / 2, h / 2)

    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    # Return the rotated image
    return rotated

def scale_resize(image, width = None, height = None, inter = cv2.INTER_CUBIC):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized


def iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    boxA = hw_to_min_max(boxA)
    boxB = hw_to_min_max(boxB)

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def filter_all_predictions(predictions, pred_threshold, iou_nms_threshold, min_bbox_square):
    result = []
    for frame in predictions:        
        new_bboxes = []
        for class_id, bboxes in enumerate(frame):
            for bbox in bboxes:
                bbox = list(bbox) + [class_id]
                if square_from_hw(min_max_to_hw(bbox[:-2])) < min_bbox_square:
                    continue
                new_bboxes.append(bbox)
        
        if len(new_bboxes) == 0:
            result.append([])
            continue

        bboxes = np.array(new_bboxes)
            
        indices = nms(bboxes[:, :-1], iou_nms_threshold)[1]
        bboxes = bboxes[indices]
            
        bboxes = bboxes[bboxes[:, -2] >= pred_threshold]
        frame_bboxes = []
        for bbox in bboxes:
            frame_bboxes.append(list(min_max_to_hw(bbox[:-2])) + [int(bbox[-1])])
                
        result.append(frame_bboxes)
    return result



def default_loader(path):
    return Image.open(path)#.convert('RGB')

def default_flist_reader(flist):
    """
    flist format: impath label\nimpath label\n ...(same to caffe's filelist)
    """
    imlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            impath, imlabel = line.strip().split()
            imlist.append( (impath, int(imlabel)) )

    return imlist

class ImageFilelist(data.Dataset):
    def __init__(self, images_data_path, flist, loader=default_loader):
        self.images_data_path = images_data_path
        self.imlist = flist
        self.loader = loader

    def __getitem__(self, index):
        impath = self.imlist[index]
        target = 0
        #img = self.loader(os.path.join(self.images_data_path, impath))
        arr_img = cv2.imread(os.path.join(self.images_data_path, impath))
        arr_img = cv2.cvtColor(arr_img, cv2.COLOR_BGR2GRAY)

        #arr_img = np.array(img.convert('RGB'))
        #img.close()
        return [arr_img, target]

    def __len__(self):
        return len(self.imlist)

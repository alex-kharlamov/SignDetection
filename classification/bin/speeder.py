import glob
import os
import os.path
import pickle
import time

import cv2
import argparse
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image

from sign_pipeline.associated import AVAILABLE_CLASSES


ALL_CLASSES_PATH = "/root/all_classes.txt"
SUBMIT_PATH = "/root/submit.tsv"
PRED_THRESHOLD = 0.65
IOU_NMS_THRESHOLD = 0.05
MIN_BBOX_SQUARE = 100
TRACKING_MIN_IOU = 0.001
TRACKING_IOU_INTERPOLATE = 1  # 0.9
TRACKING_MIN_CORRELATION = 0.3
TRACKING_MAX_DISTANCE_PIXELS = 250
BIG_CROP_PADDING = 20
TEMPORARY_THRESHOLD = 0.51 # TODO CHANGE IT
ASSOCIATED_THRESHOLD = 0.5  # TODO CHANGE IT
NUM_WORKERS = 4
POINTS = 2


def histeq(image):
    reshaped = image.reshape((image.shape[0] * image.shape[1], 3)).astype("float32")
    y = (reshaped[:, 0] * 1 + reshaped[:, 1] * 1 + reshaped[:, 2] * 2) / 4
    y = y.astype("uint8")

    hist = np.bincount(y, minlength=256).astype("float32")

    hist_sum = np.cumsum(hist)
    max_value = hist_sum[-1] / 255

    map_value = (hist_sum / max_value).astype("uint8")
    return map_value[image]


def load_img(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.cvtColor(img, cv2.COLOR_BAYER_RG2RGB)
    img = histeq(img)
    return Image.fromarray(img)


def hw_to_min_max(box):
    return list(map(float, [box[0], box[1], box[2] + box[0], box[3] + box[1]]))


def min_max_to_hw(cur_box):
    return (cur_box[0], cur_box[1], cur_box[2] - cur_box[0], cur_box[3] - cur_box[1])


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


def default_loader(path):
    return Image.open(path)  # .convert('RGB')


def default_flist_reader(flist):
    """
    flist format: impath label\nimpath label\n ...(same to caffe's filelist)
    """
    imlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            impath, imlabel = line.strip().split()
            imlist.append((impath, int(imlabel)))

    return imlist


class ImageFilelist(data.Dataset):
    def __init__(self, images_data_path, flist, loader=default_loader):
        self.images_data_path = images_data_path
        self.imlist = flist
        self.loader = loader

    def __getitem__(self, index):
        impath = self.imlist[index]
        target = 0
        # img = self.loader(os.path.join(self.images_data_path, impath))
        arr_img = cv2.imread(os.path.join(impath))
        arr_img = cv2.cvtColor(arr_img, cv2.COLOR_BGR2GRAY)

        # arr_img = np.array(img.convert('RGB'))
        # img.close()
        return [arr_img, target]

    def __len__(self):
        return len(self.imlist)


def match_bboxes_iou(start_frame_bboxes, finish_frame_bboxes):
    # Returns list of pairs (start_bbox, finish_bbox, start_bbox_index, finish_bbox_index)
    # where each bbox is a pair (bbox, class_id, temporary, associated_class_id)

    start_to_finish_iou = []
    for i, start_bbox in enumerate(start_frame_bboxes):
        for j, finish_bbox in enumerate(finish_frame_bboxes):
            score = iou(start_bbox[0], finish_bbox[0])
            if start_bbox[1] != finish_bbox[1]:
                score = 0

            start_to_finish_iou.append((score, i, j))

    start_to_finish_iou.sort(key=lambda x: x[0], reverse=True)

    matched_pairs = []
    used_start_bboxes = set()
    used_finish_bboxes = set()

    for score, i, j in start_to_finish_iou:
        if score < TRACKING_MIN_IOU:
            break

        if i in used_start_bboxes or j in used_finish_bboxes:
            continue

        start_bbox = start_frame_bboxes[i]
        finish_bbox = finish_frame_bboxes[j]
        matched_pairs.append((start_bbox, finish_bbox, i, j))

        used_start_bboxes.add(i)
        used_finish_bboxes.add(j)

    return matched_pairs


def get_distance_between_bboxes(start_bbox, finish_bbox):
    distance = (start_bbox[0] - finish_bbox[0]) ** 2 + (start_bbox[1] - finish_bbox[1]) ** 2
    return distance ** (1 / 2.)


def match_bboxes_template(start_frame, finish_frame):
    # Returns list of pairs (start_bbox, finish_bbox, start_bbox_index, finish_bbox_index)
    # where each bbox is a pair (bbox, class_id, temporary, associated_class_id)

    start_frame_bboxes = start_frame[0]
    finish_frame_bboxes = finish_frame[0]

    start_frame_img = start_frame[1]
    finish_frame_img = finish_frame[1]

    start_to_finish_iou = []
    for i, start_bbox in enumerate(start_frame_bboxes):
        for j, finish_bbox in enumerate(finish_frame_bboxes):
            if start_bbox[1] != finish_bbox[1]:
                continue

            distance = get_distance_between_bboxes(start_bbox[0], finish_bbox[0])
            if distance > TRACKING_MAX_DISTANCE_PIXELS:
                continue

            start_crop = start_frame_img[
                         int(start_bbox[0][1]):int(start_bbox[0][1] + start_bbox[0][3]),
                         int(start_bbox[0][0]):int(start_bbox[0][0] + start_bbox[0][2])]

            finish_crop = finish_frame_img[
                          int(finish_bbox[0][1]):int(finish_bbox[0][1] + finish_bbox[0][3]),
                          int(finish_bbox[0][0]):int(finish_bbox[0][0] + finish_bbox[0][2])]

            h = min(start_crop.shape[0], finish_crop.shape[0])
            w = min(start_crop.shape[1], finish_crop.shape[1])
            start_crop = cv2.resize(start_crop, (w, h))
            finish_crop = cv2.resize(finish_crop, (w, h))

            score = cv2.matchTemplate(start_crop, finish_crop, cv2.TM_CCOEFF_NORMED)[0][0]

            start_to_finish_iou.append((score, i, j))

    start_to_finish_iou.sort(key=lambda x: x[0], reverse=True)

    matched_pairs = []
    used_start_bboxes = set()
    used_finish_bboxes = set()

    for score, i, j in start_to_finish_iou:
        if score < TRACKING_MIN_CORRELATION:
            break

        if i in used_start_bboxes or j in used_finish_bboxes:
            continue

        start_bbox = start_frame_bboxes[i]
        finish_bbox = finish_frame_bboxes[j]
        matched_pairs.append((start_bbox, finish_bbox, i, j))

        used_start_bboxes.add(i)
        used_finish_bboxes.add(j)

    return matched_pairs


def match_bboxes(start_frame, finish_frame):
    # Returns list of pairs (start_bbox, finish_bbox)
    # where each bbox is a pair (bbox, class_id, temporary, associated_class_id)

    matched_pairs = []

    matched_pairs_iou = match_bboxes_iou(start_frame[0], finish_frame[0])
    start_bbox_indices_to_remove = set()
    finish_bbox_indices_to_remove = set()

    for start_bbox, finish_bbox, start_bbox_index, finish_bbox_index in matched_pairs_iou:
        start_bbox_indices_to_remove.add(start_bbox_index)
        finish_bbox_indices_to_remove.add(finish_bbox_index)

        matched_pairs.append((start_bbox, finish_bbox))

    new_start_frame_bboxes = []
    new_finish_frame_bboxes = []

    for i, bbox in enumerate(start_frame[0]):
        if i not in start_bbox_indices_to_remove:
            new_start_frame_bboxes.append(bbox)

    for j, bbox in enumerate(finish_frame[0]):
        if j not in finish_bbox_indices_to_remove:
            new_finish_frame_bboxes.append(bbox)

    start_frame = [new_start_frame_bboxes, start_frame[1]]
    finish_frame = [new_finish_frame_bboxes, finish_frame[1]]

    matched_pairs_template = match_bboxes_template(start_frame, finish_frame)

    for start_bbox, finish_bbox, start_bbox_index, finish_bbox_index in matched_pairs_template:
        matched_pairs.append((start_bbox, finish_bbox))

    return matched_pairs


def run_tracking(sequence_tracking):
    result = []

    start_frame = sequence_tracking[0]
    finish_frame = sequence_tracking[-1]

    start_frame_img = start_frame[1]
    finish_frame_img = finish_frame[1]

    interpolate_frames = sequence_tracking[1:-1]

    matched_bboxes = match_bboxes(start_frame, finish_frame)

    for frame_num, (_, image) in enumerate(interpolate_frames):
        frame_bboxes = []
        for start_bbox, finish_bbox in matched_bboxes:
            class_id = start_bbox[1]
            temporary = start_bbox[2]
            associated_class_id = start_bbox[3]

            start_bbox = start_bbox[0]
            finish_bbox = finish_bbox[0]

            start_bbox_size = [start_bbox[2], start_bbox[3]]
            finish_bbox_size = [finish_bbox[2], finish_bbox[3]]

            bbox_diff = [
                finish_bbox[0] - start_bbox[0],
                finish_bbox[1] - start_bbox[1],
                finish_bbox[2] - start_bbox[2],
                finish_bbox[3] - start_bbox[3]]

            new_coarse_bbox = start_bbox[:]
            for i in range(4):
                new_coarse_bbox[i] += bbox_diff[i] / (len(interpolate_frames) + 1) * (frame_num + 1)

            if iou(start_bbox, finish_bbox) > TRACKING_IOU_INTERPOLATE:
                frame_bboxes.append((new_coarse_bbox, class_id, temporary, associated_class_id))
                continue

            start_bbox = hw_to_min_max(start_bbox)
            finish_bbox = hw_to_min_max(finish_bbox)

            bbox_find_area = [
                min(start_bbox[0], finish_bbox[0]) - BIG_CROP_PADDING,
                min(start_bbox[1], finish_bbox[1]) - BIG_CROP_PADDING,
                max(start_bbox[2], finish_bbox[2]) + BIG_CROP_PADDING,
                max(start_bbox[3], finish_bbox[3]) + BIG_CROP_PADDING
            ]

            bbox_find_area = [
                max(bbox_find_area[0], 0),
                max(bbox_find_area[1], 0),
                min(bbox_find_area[2], image.shape[1]),
                min(bbox_find_area[3], image.shape[0]),
            ]

            crop_proposals = []

            start_crop = start_frame_img[
                         int(start_bbox[1]):int(start_bbox[3]),
                         int(start_bbox[0]):int(start_bbox[2])]
            #             crop_proposals.append((start_crop, start_bbox_size[0], start_bbox_size[1]))

            #             finish_crop = finish_frame_img[
            #                 int(finish_bbox[1]):int(finish_bbox[3]),
            #                 int(finish_bbox[0]):int(finish_bbox[2])]
            #             crop_proposals.append((finish_crop, finish_bbox_size[0], finish_bbox_size[1]))

            estimated_crop = cv2.resize(start_crop, (int(new_coarse_bbox[2]), int(new_coarse_bbox[3])))
            crop_proposals.append((estimated_crop, estimated_crop.shape[1], estimated_crop.shape[0]))

            find_area_crop = image[
                             int(bbox_find_area[1]):int(bbox_find_area[3]),
                             int(bbox_find_area[0]):int(bbox_find_area[2])]

            crop_proposals_scores = []
            for crop_proposal in crop_proposals:
                score = cv2.matchTemplate(find_area_crop, crop_proposal[0], cv2.TM_CCOEFF_NORMED)
                score_argmax = np.argmax(score)

                crop_proposals_scores.append((score_argmax, score, crop_proposal[1], crop_proposal[2]))

            best_proposal = max(crop_proposals_scores, key=lambda x: x[0])

            match_result = np.unravel_index(best_proposal[0], best_proposal[1].shape)
            new_y_min = float(match_result[0] + int(bbox_find_area[1]))
            new_x_min = float(match_result[1] + int(bbox_find_area[0]))
            new_y_max = new_y_min + best_proposal[3]
            new_x_max = new_x_min + best_proposal[2]

            new_bbox = [new_x_min, new_y_min, new_x_max, new_y_max]
            new_bbox = min_max_to_hw(new_bbox)

            frame_bboxes.append((new_bbox, class_id, temporary, associated_class_id))

        result.append(frame_bboxes)

    result.append(finish_frame[0])
    return result


def write_submit(submit_path, selected_filenames, final_boxes):
    with open(ALL_CLASSES_PATH) as f:
        all_pos_classes = f.read().split()

    convert_class = lambda x: '.'.join(str(x).split('.')[:POINTS])

    all_pos_classes = [convert_class(sign) for sign in all_pos_classes]

    mode = "w"
    if os.path.exists(submit_path):
        mode = "a"

    with open(submit_path, mode) as f:
        if mode == "w":
            f.write('\t'.join(['frame', 'xtl', 'ytl', 'xbr', 'ybr', 'class', 'temporary', 'data']) + '\n')

        for ind in range(len(selected_filenames)):
            img_name = selected_filenames[ind]
            img_name = img_name.replace('.pnm', '')
            img_name = "/".join(img_name.split("/")[-2:])
            for bbox, class_id, temporary, associated_class_id in final_boxes[ind]:
                class_id = int(class_id)
                class_name = all_pos_classes[class_id]
                bbox = hw_to_min_max(bbox)
                bbox = list(map(lambda x: str(int(x)), bbox))
                temporary = "true" if temporary == 1 else "false"
                associated_class_name = ""
                if associated_class_id < len(AVAILABLE_CLASSES):
                    associated_class_name = AVAILABLE_CLASSES[associated_class_id]

                f.write('\t'.join([img_name, *bbox, class_name, temporary, associated_class_name]) + '\n')


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("sequence_path")
    parser.add_argument("predictions_path")

    args = parser.parse_args()

    with open(args.predictions_path, 'rb') as fin:
        detector_predictions = pickle.load(fin)

    file_names = []

    for cur_img in glob.glob(args.sequence_path + "/**", recursive=True):
        if not ".pnm" in cur_img:
            continue
        file_names.append(cur_img)

    file_names = np.array(file_names)
    video_seq = np.argsort(file_names)[::-1]
    selected_filenames = file_names[video_seq]

    dataset = ImageFilelist(args.sequence_path, selected_filenames)
    loader = torch.utils.data.DataLoader(dataset,
                                         shuffle=False,
                                         num_workers=NUM_WORKERS)
    loader = iter(loader)

    final_boxes = []
    start_time = time.time()

    current_sequence_tracking = []
    current_sequence_index = 0

    for ind in range(len(selected_filenames)):
        cur_img_gray = next(loader)[0][0].data.numpy()

        frame_predictions = detector_predictions[ind]

        frame_final_boxes = []
        if frame_predictions is not None:
            frame_final_boxes = []
            for prediction in frame_predictions:
                bbox = prediction[0][:4]
                class_id = int(prediction[0][4])
                associated_class_id = prediction[1]
                associated_prob = prediction[2]
                temporary_prob = prediction[3]

                temporary = 1 if temporary_prob > TEMPORARY_THRESHOLD else 0
                associated_class_id = associated_class_id \
                    if associated_prob > ASSOCIATED_THRESHOLD else len(AVAILABLE_CLASSES)

                frame_final_boxes.append((bbox, class_id, temporary, associated_class_id))

        current_sequence_tracking.append((frame_final_boxes, cur_img_gray))

        if frame_predictions is not None and len(current_sequence_tracking) > 1:
            if current_sequence_index == 0:
                final_boxes.append(current_sequence_tracking[0][0])
                current_sequence_index += 1
            for boxes in run_tracking(current_sequence_tracking):
                final_boxes.append(boxes)

            current_sequence_tracking = [current_sequence_tracking[-1]]

        current_time = int(time.time() - start_time)
        time_per_iter = (time.time() - start_time) / (ind + 1)
        eta_time = int((len(selected_filenames) - ind) * time_per_iter)
        print("\rIndex: {}. Time: {} s. ETA: {} s".format(ind, current_time, eta_time), end="")

    if len(current_sequence_tracking) > 0:
        for boxes, _ in current_sequence_tracking[1:]:
            final_boxes.append(boxes)

    write_submit(SUBMIT_PATH, selected_filenames, final_boxes)


if __name__ == "__main__":
    main()

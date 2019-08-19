import argparse
import pickle

import numpy as np
from mmdet.ops.nms import nms


PRED_THRESHOLD = 0.55
IOU_NMS_THRESHOLD = 0.05
MIN_BBOX_SQUARE = 100


def min_max_to_hw(cur_box):
    return cur_box[0], cur_box[1], cur_box[2] - cur_box[0], cur_box[3] - cur_box[1]


def square_from_hw(cur_box):
    return cur_box[2] * cur_box[3]


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions_path")
    parser.add_argument("output_path")

    args = parser.parse_args()

    with open(args.predictions_path, 'rb') as fin:
        detector_predictions = filter_all_predictions(
            pickle.load(fin),
            PRED_THRESHOLD,
            IOU_NMS_THRESHOLD,
            MIN_BBOX_SQUARE)

    with open(args.output_path, "wb") as fout:
        pickle.dump(detector_predictions, fout)


if __name__ == "__main__":
    main()

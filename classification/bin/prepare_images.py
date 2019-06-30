import argparse
import pickle

import tqdm
from PIL import Image
import random

NUM_EMPTY = 1
EMPTY_LABEL_ID = -1
EMPTY_MIN_SIZE = 20
EMPTY_MAX_SIZE = 100
MAX_FIND_EMPTY_BBOX_ITER = 10


def is_bounding_boxes_intersect(bbox_one, bbox_second):
    if bbox_one[2] <= bbox_second[0] or bbox_one[0] >= bbox_second[2] or \
            bbox_one[3] <= bbox_second[1] or bbox_one[1] >= bbox_second[3]:
        return False
    return True


def sample_empty_bbox(img_size):
    x_side = random.randint(EMPTY_MIN_SIZE, EMPTY_MAX_SIZE)
    y_side = random.randint(EMPTY_MIN_SIZE, EMPTY_MAX_SIZE)

    x_start = random.randint(0, img_size[0] - x_side - 1)
    y_start = random.randint(0, img_size[1] - y_side - 1)

    return [x_start, y_start, x_start + x_side, y_start + y_side]


def find_empty_bbox(bboxes, img_size):
    for _ in range(MAX_FIND_EMPTY_BBOX_ITER):
        bbox = sample_empty_bbox(img_size)
        is_intersect = False

        for another_bbox in bboxes:
            if is_bounding_boxes_intersect(bbox, another_bbox):
                is_intersect = True
                break

        if is_intersect:
            continue

        return bbox

    return None


def crop_image(image, bbox):
    return image.crop(bbox)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("annotations_path")
    parser.add_argument("output_path")

    args = parser.parse_args()

    with open(args.annotations_path, "rb") as fin:
        annotations = pickle.load(fin)

    result = []

    for annotation in tqdm.tqdm(annotations):
        image = Image.open(annotation["filename"]).convert("RGB")

        bboxes = annotation["ann"]["bboxes"]
        labels = annotation["ann"]["labels"]

        for bbox, label in zip(bboxes, labels):
            data = {
                "bbox": bbox,
                "label": label,
                "cropped_image": crop_image(image, bbox)
            }

            result.append(data)

        for _ in range(NUM_EMPTY):
            bbox = find_empty_bbox(bboxes, image.size)
            if bbox is None:
                break

            data = {
                "bbox": bbox,
                "label": EMPTY_LABEL_ID,
                "cropped_image": crop_image(image, bbox)
            }

            result.append(data)

        if len(result) > 1000:
            break

    with open(args.output_path, "wb") as fout:
        pickle.dump(result, fout)


if __name__ == "__main__":
    main()

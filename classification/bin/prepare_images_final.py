import argparse
import os
import pickle

import cv2
import joblib
from PIL import Image
from colour_demosaicing import demosaicing_CFA_Bayer_bilinear

from sign_pipeline.associated import AVAILABLE_CLASSES

N_JOBS = 16
PADDING_PERCENT = 20


def load_img(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    image = demosaicing_CFA_Bayer_bilinear(image).astype("uint8")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image)


def crop_image(image, bbox):
    padding_x = int((bbox[2] - bbox[0]) / 100. * PADDING_PERCENT)
    padding_y = int((bbox[3] - bbox[1]) / 100. * PADDING_PERCENT)
    bbox = [bbox[0] - padding_x, bbox[1] - padding_y, bbox[2] + padding_x, bbox[3] + padding_y]
    bbox[0] = max(bbox[0], 0)
    bbox[1] = max(bbox[1], 0)
    bbox[2] = min(bbox[2], image.size[0])
    bbox[3] = min(bbox[3], image.size[1])

    return image.crop(bbox)


def extract_annotations(data_path, annotation):
    result = []
    image = load_img(os.path.join(data_path, annotation["filename"]))

    bboxes = annotation["ann"]["bboxes"]
    labels = annotation["ann"]["labels"]
    temporary_labels = annotation["ann"]["temporary"]
    associated_data = annotation["ann"]["data"]

    for bbox, label, temporary, data in zip(bboxes, labels, temporary_labels, associated_data):
        if data in AVAILABLE_CLASSES:
            associated_label_id = AVAILABLE_CLASSES.index(data)
        else:
            associated_label_id = len(AVAILABLE_CLASSES)

        data = {
            "bbox": bbox,
            "temporary": int(temporary),
            "associated_label": associated_label_id,
            "cropped_image": crop_image(image, bbox)
        }

        result.append(data)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path")
    parser.add_argument("annotations_path")
    parser.add_argument("output_path")

    args = parser.parse_args()

    with open(args.annotations_path, "rb") as fin:
        annotations = pickle.load(fin)

    p = joblib.Parallel(n_jobs=N_JOBS, backend="multiprocessing", verbose=5)
    extracted_annotations = p(joblib.delayed(extract_annotations)(args.data_path, annotation) for annotation in annotations)
    result = []

    for annotation in extracted_annotations:
        result += annotation

    with open(args.output_path, "wb") as fout:
        pickle.dump(result, fout)


if __name__ == "__main__":
    main()

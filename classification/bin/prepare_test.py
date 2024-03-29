import argparse
import pickle

import joblib
from PIL import Image
import cv2
import numpy as np


N_JOBS = 16


def crop_image(image, bbox):
    return image.crop(bbox)


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


def extract_bboxes(annotation, bboxes):
    result = []
    image = load_img(annotation["filename"])

    for bbox in bboxes:
        data = {
            "bbox": bbox,
            "cropped_image": crop_image(
                image, [int(bbox[0]), int(bbox[1]), int(bbox[2] + bbox[0]), int(bbox[3] + bbox[1])]),
            "filename": annotation["filename"]
        }

        result.append(data)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("annotations_path")
    parser.add_argument("bboxes_predict_path")
    parser.add_argument("output_path")

    args = parser.parse_args()

    with open(args.annotations_path, "rb") as fin:
        annotations = pickle.load(fin)

    with open(args.bboxes_predict_path, "rb") as fin:
        bboxes_predict = pickle.load(fin)

    p = joblib.Parallel(n_jobs=N_JOBS, backend="multiprocessing", verbose=5)
    extracted_bboxes = p(
        joblib.delayed(extract_bboxes)(annotation, bboxes) for annotation, bboxes in zip(annotations, bboxes_predict))

    result = []
    for bboxes in extracted_bboxes:
        result += bboxes

    with open(args.output_path, "wb") as fout:
        pickle.dump(result, fout)


if __name__ == "__main__":
    main()

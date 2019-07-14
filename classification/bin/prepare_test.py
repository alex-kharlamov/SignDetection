import argparse
import pickle

import joblib
from PIL import Image
import cv2
from colour_demosaicing import demosaicing_CFA_Bayer_bilinear


N_JOBS = 16


def crop_image(image, bbox):
    return image.crop(bbox)


def load_img_new(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    image = demosaicing_CFA_Bayer_bilinear(image).astype("uint8")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image)

def load_img(path):
    path = path.replace("/group-volume/orc_srr/multimodal/iceblood/datasets/main/", "/Vol1/dbstore/datasets/multimodal/iceblood/")
    return Image.open(path)


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

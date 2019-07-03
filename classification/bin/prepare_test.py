import argparse
import pickle

import joblib
from PIL import Image

N_JOBS = 16


def crop_image(image, bbox):
    return image.crop(bbox)


def extract_bboxes(annotation, bboxes):
    result = []
    image = Image.open(annotation["filename"]).convert("RGB")

    for class_bboxes in bboxes:
        for bbox in class_bboxes:
            if len(bbox) == 0:
                continue
            bbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
            data = {
                "bbox": bbox,
                "cropped_image": crop_image(image, bbox),
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

import argparse
import torch
import pickle


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("predictions_path")
    parser.add_argument("extracted_bboxes_path")
    parser.add_argument("labels_mapping_path")
    parser.add_argument("output_path")

    args = parser.parse_args()

    predictions = torch.load(args.predictions_path)

    with open(args.extracted_bboxes_path, "rb") as fin:
        extracted_bboxes = pickle.load(fin)

    with open(args.labels_mapping_path, "rb") as fin:
        labels_mapping = pickle.load(fin)

    labels_reverse_mapping = {}
    for key, value in labels_mapping.items():
        labels_reverse_mapping[value] = key

    result = []
    for prediction, info in zip(predictions, extracted_bboxes):
        prediction = prediction.cpu()
        prediction = torch.softmax(prediction, dim=0)
        label_id = int(prediction.argmax())
        probability = float(prediction.max())

        label = labels_reverse_mapping[label_id]
        result.append([info["filename"], info["bbox"], label, probability])

    with open(args.output_path, "wb") as fout:
        pickle.dump(result, fout)


if __name__ == "__main__":
    main()

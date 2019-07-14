import argparse
import torch
import pickle


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("predictions_path")
    parser.add_argument("extracted_bboxes_path")
    parser.add_argument("output_path")

    args = parser.parse_args()

    predictions = torch.load(args.predictions_path)

    with open(args.extracted_bboxes_path, "rb") as fin:
        extracted_bboxes = pickle.load(fin)

    result = []
    for prediction, info in zip(predictions, extracted_bboxes):
        prediction = prediction.cpu()

        prediction_multi = prediction[:-1]
        prediction_binary = prediction[-1]

        label_multi = prediction_multi.argmax()

        result.append([info["filename"], info["bbox"], label_multi, torch.sigmoid(prediction_binary)])

    with open(args.output_path, "wb") as fout:
        pickle.dump(result, fout)


if __name__ == "__main__":
    main()

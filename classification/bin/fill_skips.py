import argparse
import os
import pickle
import copy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sequence_path")
    parser.add_argument("predictions_path")
    parser.add_argument("output_path")

    parser.add_argument("--skip", type=int, default=1)

    args = parser.parse_args()

    with open(args.predictions_path, "rb") as fin:
        predictions = pickle.load(fin)

    index = 0
    predict_index = 0
    result = []
    files = sorted(os.listdir(args.sequence_path), reverse=True)
    files = list(filter(lambda x: x.endswith(".pnm"), files))

    for file_name in files:
        index += 1
        if (index - 1) % args.skip != 0 and file_name != files[-1]:
            result.append(None)
            continue

        result.append(predictions[predict_index])
        predict_index += 1

    with open(args.output_path, "wb") as fout:
        pickle.dump(result, fout)


if __name__ == "__main__":
    main()


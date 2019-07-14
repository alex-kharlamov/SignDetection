import argparse
import os
import pickle
import copy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sequence_path")
    parser.add_argument("test_pickle_path")
    parser.add_argument("output_path")

    parser.add_argument("--skip", type=int, default=1)

    args = parser.parse_args()

    with open(args.test_pickle_path, "rb") as fin:
        pickle_row = pickle.load(fin)[0]

    sequence_rows = []
    index = 0
    files = sorted(os.listdir(args.sequence_path), reverse=True)
    files = list(filter(lambda x: x.endswith(".pnm"), files))
    for file_name in files:
        index += 1
        if (index - 1) % args.skip != 0 and file_name != files[-1]:
            continue

        row = copy.deepcopy(pickle_row)
        row["filename"] = os.path.join(args.sequence_path, file_name)
        sequence_rows.append(row)

    with open(args.output_path, "wb") as fout:
        pickle.dump(sequence_rows, fout)


if __name__ == "__main__":
    main()


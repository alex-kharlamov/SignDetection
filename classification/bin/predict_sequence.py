import argparse
import os
import subprocess

TMP_PATH = ""
TEST_PICKLE_PATH = ""
SKIP_FRAMES_NUM = 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sequence_path")

    args = parser.parse_args()

    os.makedirs(TMP_PATH, exist_ok=True)

    print("Starting prediction sequence", args.sequence_path)

    print("Running prepare_test_annotations.py...")
    subprocess.check_call([
        "python3",
        "prepare_test_annotations.py",
        args.sequence_path,
        TEST_PICKLE_PATH,
        os.path.join(TMP_PATH, "annotations.pickle"),
        "--skip",
        str(SKIP_FRAMES_NUM)])

    print("Running mmdet...")
    """
    ./tools/test.py configs/fp16_101_full_annotations_predict.py work_dirs/fp32_cascade_rcnn_x100_64x4d_fpn_1x_fit/cascade_vmk_pretrain.pth --out result/fp16_101_annotation_predict.pkl
    """
    # TODO! mmdet prediction command

    print("Running filter_predictions.py...")
    subprocess.check_call([
        "python3",
        "filter_predictions.py",
        os.path.join(TMP_PATH, "detector_output.pickle"),
        os.path.join(TMP_PATH, "detector_filtered.pickle")])

    print("Running prepare_test.py...")
    subprocess.check_call([
        "python3",
        "prepare_test.py",
        os.path.join(TMP_PATH, "annotations.pickle"),
        os.path.join(TMP_PATH, "detector_filtered.pickle"),
        os.path.join(TMP_PATH, "classificator_input.pickle")])

    print("Running predict.py...")  # TODO! run classificator predict correctly
    subprocess.check_call([
        "python3",
        "predict.py", ])

    print("Running construct_predictions.py...")
    subprocess.check_call([
        "python3",
        "construct_predictions.py",
        # TODO! classificator predictions path,
        os.path.join(TMP_PATH, "classificator_input.pickle"),
        os.path.join(TMP_PATH, "classificator_output.pickle")])

    print("Running fill_skips.py...")
    subprocess.check_call([
        "python3",
        "fill_skips.py",
        args.sequence_path,
        os.path.join(TMP_PATH, "classificator_output.pickle"),
        os.path.join(TMP_PATH, "speeder_input.pickle")])

    print("Running speeder.py...")
    subprocess.check_call([
        "python3",
        "speeder.py",
        args.sequence_path,
        os.path.join(TMP_PATH, "speeder_input.pickle")])


if __name__ == "__main__":
    main()

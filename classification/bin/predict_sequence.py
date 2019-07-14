import argparse
import os
import subprocess
import shutil

TMP_PATH = "/root/SignDetection/classification/bin/tmp_folder"
TEST_PICKLE_PATH = "/root/SignDetection/classification/bin/tmp_folder/test.pickle"
SKIP_FRAMES_NUM = 1
CLASSIFICATOR_CONFIG = "/root/SignDetection/classification/sign_pipeline/configs/resnet34.py"
CLASSIFICATOR_PREDICTIONS_FOLDER = "/root/SignDetection/classification/models/resnet34/predictions"


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

    if os.path.exists(CLASSIFICATOR_PREDICTIONS_FOLDER):
        shutil.rmtree(CLASSIFICATOR_PREDICTIONS_FOLDER)

    print("Running predict.py...")
    subprocess.check_call([
        "python3",
        "predict.py",
        CLASSIFICATOR_CONFIG])

    print("Running construct_predictions.py...")
    subprocess.check_call([
        "python3",
        "construct_predictions.py",
        os.path.join(CLASSIFICATOR_PREDICTIONS_FOLDER, "predictions"),
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

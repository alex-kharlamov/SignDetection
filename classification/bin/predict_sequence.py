import argparse
import os
import subprocess
import shutil

TMP_PATH = "/root/SignDetection/classification/bin/tmp_folder"
TEST_PICKLE_PATH = "/root/SignDetection/classification/bin/tmp_folder/test.pickle"
SKIP_FRAMES_NUM = 8
CLASSIFICATOR_CONFIG = "sign_pipeline.configs.resnet34.py"
CLASSIFICATOR_PREDICTIONS_FOLDER = "/root/SignDetection/classification/bin/models/sign_resnet34/predictions"

MMDETECTION_BINARY_PATH = "/root/mmsetection_pnm/tools/test.py"
MMDETECTION_CONFIG_PATH = "/root/fp16_cascade_rcnn_50_sk_fit_predict_085.py"
MMDETECTION_CHECKPOINT_PATH = "/root/our_data/916_fp16_cascade_rcnn_x50_32x4d_fpn_1x_fit_85_epoch_27.pth"


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
    subprocess.check_call([
        "python3",
        MMDETECTION_BINARY_PATH,
        MMDETECTION_CONFIG_PATH,
        MMDETECTION_CHECKPOINT_PATH,
        "--out",
        os.path.join(TMP_PATH, "detector_output.pickle")])

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

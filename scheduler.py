import os
import numpy as np
import pickle
from shutil import copyfile
import glob
import time
import itertools
import shutil
from joblib import Parallel, delayed
import subprocess
import argparse


IMGS_STORED = 3
work_dir = 'work_dir/'
COPER_FOLDER = 2
PARALLEL_COPY_WORKERS = 4



parser = argparse.ArgumentParser()
parser.add_argument("test_folder")
args = parser.parse_args()


folders = glob.glob(args.test_folder + '/*/*', recursive=True)

all_batches = []
for cur_folder in folders:
    internal_files = sorted(glob.glob(os.path.join(cur_folder, '*'), recursive=True))
    for ind in range(0, len(internal_files), IMGS_STORED):
        all_batches.append(internal_files[ind:ind + IMGS_STORED])


def _check_existance(cur_fold, status):
    if os.path.exists(os.path.join(cur_fold, status)):
        return True
    else:
        return False

def check_transfered(cur_fold):
    return _check_existance(cur_fold, 'transfered')
    
def check_predicted(cur_fold):
    return _check_existance(cur_fold, 'predicted')
    
def clean_dir(cur_fold):
    shutil.rmtree(cur_fold)
    os.makedirs(cur_fold, exist_ok=True)

    
def _single_copy_file(arg):
    source, cur_fold = arg
    folder = os.path.dirname(source)
    base_folders = os.path.dirname(source).split('/')[-2:]
    base_folders = '_'.join(base_folders)
    name = os.path.basename(source)
    #print(os.path.join(cur_fold, base_folder, name))
    os.makedirs(os.path.join(cur_fold, base_folders), exist_ok=True)
    copyfile(source, os.path.join(cur_fold, base_folders, name))
    

def copy_files(cur_fold, imgs):
    arguments = [(cur_img, cur_fold) for cur_img in imgs]
    results = Parallel(n_jobs=PARALLEL_COPY_WORKERS,
                       verbose=1,
                       backend="threading")(map(delayed(_single_copy_file), arguments))
    time.sleep(5)

def set_transfered(cur_fold):
    with open(os.path.join(cur_fold, 'transfered'), 'w') as f:
        f.write('transfered')

def set_predicted(cur_fold):
    with open(os.path.join(cur_fold, 'predicted'), 'w') as f:
        f.write('predicted')

def _background_inderence_folder(folder):
    #process = subprocess.Popen(["python3", "predict_sequence.py", folder])
    process = subprocess.Popen(["sleep", "1"])
    return process

def check_process(process):
    try:
        process.wait(timeout=0.001)
    except subprocess.TimeoutExpired:
        return False
    return True
        
def get_prediction(cur_fold):
    for cand_fold in os.listdir(cur_fold):
        full_path = os.path.join(cur_fold, cand_fold)
        if os.path.isdir(full_path):
            process = _background_inderence_folder(full_path)
            return process


folders = [work_dir + str(worker) for worker in range(COPER_FOLDER)]
for cur_fold in folders:
    os.makedirs(cur_fold, exist_ok=True)
    
pred_status = dict()
now_predicting = False

for img_ind in range(0, len(imgs), IMGS_STORED):
    current_imgs = imgs[img_ind: img_ind + IMGS_STORED]
    
    for cur_fold in itertools.cycle(folders):
        if cur_fold in pred_status and check_process(pred_status[cur_fold]): #Process return predicted status
            set_predicted(cur_fold)
            now_predicting = False
            clean_dir(cur_fold)
            del pred_status[cur_fold]
            print('Prediction finished on {}'.format(cur_fold))
            continue
        
        
        if not check_transfered(cur_fold): #Empty or predicted
            print('Starting copy from {} to {} on {}'.format(current_imgs[0], current_imgs[-1], cur_fold))
            copy_files(cur_fold, current_imgs)
            set_transfered(cur_fold)
            print('Copy suceeded')
        
        if not now_predicting:
            process = get_prediction(cur_fold)
            pred_status[cur_fold] = process
            now_predicting = True
            print('Prediction started on {}'.format(cur_fold))
            break #should dispatch new images
               
        time.sleep(1)
print('All predictions done')
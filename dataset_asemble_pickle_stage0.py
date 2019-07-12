import glob
import os
import pickle
from random import sample

import numpy as np
import p_tqdm
import pandas as pd
from PIL import Image
from tqdm import tqdm

save_path = '/group-volume/orc_srr/multimodal/iceblood/datasets/main/TRAIN/'
save_path_val = '/group-volume/orc_srr/multimodal/iceblood/datasets/main/TEST/'

datasets_list = ['/group-volume/orc_srr/multimodal/iceblood/datasets/main/TRAIN/skolkovo/',
                 '/group-volume/orc_srr/multimodal/iceblood/datasets/main/TRAIN/RTSD/',
                 '/group-volume/orc_srr/multimodal/iceblood/datasets/main/TEST/']

class_precision = 3
sign_min_size = 10 * 10

convert_class = lambda x: '.'.join(str(x).split('.')[:class_precision])

datasets_dataframes = []

for dataset_path in datasets_list:
    df = pd.read_csv(glob.glob(os.path.join(dataset_path, '*.csv'))[0])
    df['path'] = df['path'].apply(lambda x: os.path.join(dataset_path, x))
    df['class'] = df['class'].apply(lambda x: str(x))
    datasets_dataframes.append(df)
    print(df.shape)
    print(df['path'].nunique())

result_df = pd.concat(datasets_dataframes)
result_df = result_df.reset_index()

all_classes = [convert_class(x) for x in sorted(result_df['class'].unique())]

with open('temp_all_classes.txt', 'w') as f:
    for item in all_classes:
        f.write(item)
        f.write('\n')

result_df = result_df.reindex(sorted(result_df.columns), axis=1)


def df_to_annontation(path, df, classes):
    frame_df = df[df['path'] == path]
    annotation = {}

    if os.path.exists(path):
        im = Image.open(path)
    else:
        return None

    width, height = im.size

    annotation['filename'] = path
    annotation['width'] = width
    annotation['height'] = height
    annotation['ann'] = {}
    bboxes = []
    labels = []
    temporary = []
    occluded = []
    ignored_bboxes = []
    ignored_labels = []
    ignored_temporary = []
    ignored_occluded = []
    for index, row in frame_df.iterrows():
        cur_cls = row['class']
        xbr = row['xbr']
        xtl = row['xtl']
        ybr = row['ybr']
        ytl = row['ytl']
        temp = row['temporary']
        ocl = row['occluded']
        cur_cls = convert_class(cur_cls)
        if cur_cls in classes:
            if ((ybr - ytl) * (xbr - xtl) >= sign_min_size):
                labels.append(classes.index(cur_cls) + 1)
                bboxes.append([xtl, ytl, xbr, ybr])
                temporary.append(temp)
                occluded.append(ocl)
            else:
                ignored_labels.append(classes.index(cur_cls) + 1)
                ignored_bboxes.append([xtl, ytl, xbr, ybr])
                ignored_temporary.append(temp)
                ignored_occluded.append(ocl)

    annotation['ann']['bboxes'] = np.array(bboxes).astype(np.float32)
    annotation['ann']['labels'] = np.array(labels).astype(np.int64)
    annotation['ann']['temporary'] = np.array(temporary).astype(np.float32)
    annotation['ann']['occluded'] = np.array(occluded).astype(np.float32)

    annotation['ann']['bboxes_ignore'] = np.array(ignored_bboxes).astype(np.float32)
    annotation['ann']['labels_ignore'] = np.array(ignored_labels).astype(np.int64)
    annotation['ann']['temporary_ignore'] = np.array(ignored_temporary).astype(np.float32)
    annotation['ann']['occluded_ignore'] = np.array(ignored_occluded).astype(np.float32)

    if len(bboxes):
        return annotation


annotations = p_tqdm.p_map(lambda x: df_to_annontation(x, result_df, all_classes),
                           list(result_df['path'].unique()))

annotations_skolkovo = list(filter(None, annotations[:4915]))
annotations_vmk = list(filter(None, annotations[4915:64103]))
annotations_final = list(filter(None, annotations[64103:]))

with open(os.path.join(save_path, 'first_part_skolkovo.pickle'), 'wb') as handle:
    pickle.dump(annotations_skolkovo, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(os.path.join(save_path, 'full_russia_vmk.pickle'), 'wb') as handle:
    pickle.dump(annotations_vmk, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(os.path.join(save_path_val, 'final_skolkovo.pickle'), 'wb') as handle:
    pickle.dump(annotations_final, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(os.path.join(save_path_val, 'final_val.pickle'), 'wb') as handle:
    pickle.dump(sample(annotations_final, 100), handle, protocol=pickle.HIGHEST_PROTOCOL)


def bin_ann(ann):
    ann['ann']['labels'] = np.array([1 if x > 0 else 0 for x in ann['ann']['labels']])


pickle_out = open(os.path.join(save_path, 'first_part_skolkovo.pickle'), "rb")
first_part_skolkovo1 = pickle.load(pickle_out)

pickle_out = open(os.path.join(save_path, 'full_russia_vmk.pickle'), "rb")
full_russia_vmk1 = pickle.load(pickle_out)

pickle_out = open(os.path.join(save_path_val, 'final_skolkovo.pickle'), "rb")
final_skolkovo1 = pickle.load(pickle_out)

pickle_out = open(os.path.join(save_path_val, 'final_val.pickle'), "rb")
final_val1 = pickle.load(pickle_out)

_ = [bin_ann(x) for x in full_russia_vmk1]
_ = [bin_ann(x) for x in first_part_skolkovo1]
_ = [bin_ann(x) for x in final_val1]
_ = [bin_ann(x) for x in final_skolkovo1]

with open(os.path.join(save_path, 'first_part_skolkovo_bin.pickle'), 'wb') as handle:
    pickle.dump(first_part_skolkovo1, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(os.path.join(save_path, 'full_russia_vmk_bin.pickle'), 'wb') as handle:
    pickle.dump(full_russia_vmk1, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(os.path.join(save_path_val, 'final_skolkovo_bin.pickle'), 'wb') as handle:
    pickle.dump(final_skolkovo1, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(os.path.join(save_path_val, 'final_val_bin.pickle'), 'wb') as handle:
    pickle.dump(final_val1, handle, protocol=pickle.HIGHEST_PROTOCOL)

imgs = glob.glob('/group-volume/orc_srr/multimodal/iceblood/datasets/main/TEST/**', recursive=True)
imgs = sorted(list(filter(lambda x: '.jpg' in x, imgs)))[::-1]

all_anno_custom_test = []

for cur_img in tqdm(imgs):
    # im = Image.open(cur_img)
    cur_img = '/'.join(cur_img.split('/')[-3:])

    tmp_dice = {}
    tmp_dice['filename'] = '/group-volume/orc_srr/multimodal/iceblood/datasets/main/' + cur_img
    # im = Image.open(cur_img)
    # width, height = im.size
    tmp_dice['width'] = 2448  # width
    tmp_dice['height'] = 2048  # height
    tmp_dice['ann'] = {}

    bboxes = []
    labels = []
    ignored_bboxes = []
    ignored_labels = []

    tmp_dice['ann']['bboxes'] = np.array(bboxes).astype(np.float32)
    tmp_dice['ann']['labels'] = np.array(labels).astype(np.int64)
    tmp_dice['ann']['bboxes_ignore'] = np.array(ignored_bboxes).astype(np.float32)
    tmp_dice['ann']['labels_ignore'] = np.array(ignored_labels).astype(np.int64)

    all_anno_custom_test.append(tmp_dice)

with open('/group-volume/orc_srr/multimodal/iceblood/datasets/main/TEST/skolkovo_test_file.pickle', 'wb') as outfile:
    pickle.dump(all_anno_custom_test, outfile)

import os

import pandas as pd
from tqdm import tqdm

train_annotations = '/group-volume/orc_srr/multimodal/iceblood/datasets/main/annotations/training/'
test_annotations = '/group-volume/orc_srr/multimodal/iceblood/datasets/main/annotations/test/'
final_annotations = '/group-volume/orc_srr/multimodal/iceblood/datasets/main/annotations/final/'

data_path = '/group-volume/orc_srr/multimodal/iceblood/datasets/main/TRAIN/skolkovo/'
data_path_final = '/group-volume/orc_srr/multimodal/iceblood/datasets/main/TEST/'

folders_train = os.listdir(train_annotations)
folders_test = os.listdir(test_annotations)
folders_final = os.listdir(final_annotations)

train_data = []

for cur_folder in tqdm(folders_train):
    anno = list(filter(lambda x: 'tsv' in x, os.listdir(os.path.join(train_annotations, cur_folder))))
    imgs = list(map(lambda x: x.replace('.tsv', '.jpg'), anno))
    for cur_anno in anno:
        img_data = pd.read_csv(os.path.join(train_annotations, cur_folder, cur_anno), sep='\t')
        img_data['path'] = str(os.path.join(cur_folder, cur_anno.replace('.tsv', '.jpg')))
        train_data.append(img_data)


test_data = []

for cur_folder in tqdm(folders_test):
    anno = list(filter(lambda x: 'tsv' in x, os.listdir(os.path.join(test_annotations, cur_folder))))
    imgs = list(map(lambda x: x.replace('.tsv', '.jpg'), anno))
    for cur_anno in anno:
        img_data = pd.read_csv(os.path.join(test_annotations, cur_folder, cur_anno), sep='\t')
        img_data['path'] = str(os.path.join(cur_folder, cur_anno.replace('.tsv', '.jpg')))
        test_data.append(img_data)

final_data = []

for cur_folder in tqdm(folders_final):
    anno = list(filter(lambda x: 'tsv' in x, os.listdir(os.path.join(final_annotations, cur_folder))))
    imgs = list(map(lambda x: x.replace('.tsv', '.jpg'), anno))
    for cur_anno in anno:
        img_data = pd.read_csv(os.path.join(final_annotations, cur_folder, cur_anno), sep='\t')
        img_data['path'] = str(os.path.join(cur_folder, cur_anno.replace('.tsv', '.jpg')))
        final_data.append(img_data)


train_df = pd.concat(train_data)
test_df = pd.concat(test_data)

train_df = train_df.reset_index()
test_df = test_df.reset_index()


df = pd.concat((test_df, train_df))
df['class'] = df['class'].astype(str)
df = df.reset_index()
df = df.drop(['level_0', 'data', 'index'], axis=1)
df.to_csv(os.path.join(data_path, 'skolkovo_ann.csv'), index=False)

final_df = pd.concat(final_data)

final_df = final_df.reset_index()

final_df['class'] = final_df['class'].astype(str)
final_df = final_df.drop(['data', 'index'], axis=1)
final_df.to_csv(os.path.join(data_path_final, 'skolkovo_final.csv'), index=False)



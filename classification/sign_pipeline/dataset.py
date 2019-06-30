import torch.utils.data as data
import pickle


def is_bounding_boxes_intersect(bbox_one, bbox_second):
    if bbox_one[2] <= bbox_second[0] or bbox_one[0] >= bbox_second[2] or \
            bbox_one[3] <= bbox_second[1] or bbox_one[1] >= bbox_second[3]:
        return False
    return True


class SignDataset(data.Dataset):
    def __init__(self, path):
        with open(path, "rb") as fin:
            annotations = pickle.load(fin)

        self._data = []

        for annotation in annotations:
            bboxes = annotation["ann"]["bboxes"]
            labels = annotation["ann"]["labels"]

            for bbox, label in zip(bboxes, labels):
                result = {
                    "filename": annotation["filename"],
                    "bbox": bbox,
                    "label": label
                }

                self._data.append(result)



    def get_image(self, item):
        return self._dataset[item][0]

    def get_class(self, item):
        return self._dataset[item][1]

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, item):
        return self._dataset[item]


class CIFARImagesDataset(CIFARDataset):
    def __getitem__(self, item):
        return self.get_image(item)


class CIFARTargetsDataset(CIFARDataset):
    def __getitem__(self, item):
        return self.get_class(item)

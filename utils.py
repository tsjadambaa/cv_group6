import os

import torch
from imageio import imread
import numpy as np
from torch.utils import data


def get_flying_chairs_data_paths(root: str):
    """
    :param root: the absolute path to where dataset is located (<sample_id>_img1.ppm <sample_id>_img2.ppm <sample_id>_flow.flo)
    """

    samples = []
    for name in sorted(os.listdir(root)):
        if name.endswith('_flow.flo'):
            sample_id = name[: -9]
            img1 = os.path.join(root, sample_id + "_img1.ppm")
            img2 = os.path.join(root, sample_id + "_img2.ppm")
            flow = os.path.join(root, name)
            samples.append([[img1, img2], flow])

    return samples


def flying_chairs_loader(sample):
    inputs, target = sample[0], sample[1]
    img1, img2 = np.asarray(imread(inputs[0]), dtype=np.float32), np.asarray(imread(inputs[1]), dtype=np.float32)

    with open(target, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        assert (202021.25 == magic), 'Magic number incorrect. Invalid .flo file'
        h = np.fromfile(f, np.int32, count=1)[0]
        w = np.fromfile(f, np.int32, count=1)[0]
        data = np.fromfile(f, np.float32, count=2 * w * h)
    # Reshape data into 3D array (columns, rows, bands)
    data2D = np.resize(data, (w, h, 2))
    return [img1, img2], data2D


def make_dataset_split(dataset: [], split: float = 0.8, random_seed: int = 42):
    num_train_samples = int(len(dataset) * split)
    num_val_samples = len(dataset) - num_train_samples

    seed = torch.Generator().manual_seed(random_seed)
    return data.random_split(dataset, [num_train_samples, num_val_samples], generator=seed)


def mean_epe(input_flow, target_flow):
    return torch.norm(target_flow - input_flow, p=2, dim=1).mean()

import os
import torch
import numpy as np
from PIL import Image


class lowlight_dataset(torch.utils.data.Dataset):
    def __init__(self, images_name, dataset_folder, ):
        self.image_name_list = images_name
        self.size = 256
        self.dataset_folder = dataset_folder

    def __getitem__(self, index):
        data_lowlight = Image.open(os.path.join(self.dataset_folder, 'low', self.image_name_list[index]))
        data_highlight = Image.open(os.path.join(self.dataset_folder, 'high', self.image_name_list[index]))

        data_lowlight = data_lowlight.resize((self.size, self.size), Image.LANCZOS)
        data_highlight = data_highlight.resize((self.size, self.size), Image.LANCZOS)

        data_lowlight = (np.asarray(data_lowlight) / 255.0)
        data_lowlight = torch.from_numpy(data_lowlight).float()
        data_lowlight = data_lowlight.permute(2, 0, 1).cuda()

        data_highlight = (np.asarray(data_highlight) / 255.0)
        data_highlight = torch.from_numpy(data_highlight).float()
        data_highlight = data_highlight.permute(2, 0, 1).cuda()

        return data_lowlight, data_highlight

    def __len__(self):
        return len(self.image_name_list)

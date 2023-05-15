import os
import torch
from torch.utils.data import Dataset
from turbojpeg import TurboJPEG, TJPF_GRAY
from typing import List
from .cvtransforms import *

jpeg = TurboJPEG()


def load_labels() -> List[str]:
    with open('label_sorted.txt') as f:
        labels = f.read().splitlines()
    return labels


class LRWDataset(Dataset):
    def __init__(self, phase: str):
        self.labels = load_labels()
        self.list = []
        self.phase = phase

        for label in self.labels:
            label_dir = os.path.join('lrw_roi_npy_gray_pkl_jpeg', label, phase)
            files = [os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith('.pkl')]
            files.sort()
            self.list.extend(files)

    def __getitem__(self, idx):

        tensor = torch.load(self.list[idx])
        inputs = tensor.get('video')
        inputs = [jpeg.decode(img, pixel_format=TJPF_GRAY) for img in inputs]
        inputs = np.stack(inputs, 0) / 255.0
        inputs = inputs[:, :, :, 0]

        if self.phase == 'train':
            batch_img = random_crop(inputs, (88, 88))
            batch_img = horizontal_flip(batch_img)
        elif self.phase == 'val' or self.phase == 'test':
            batch_img = center_crop(inputs, (88, 88))

        result = {
            'video': torch.FloatTensor(batch_img[:, np.newaxis, ...]),
            'label': tensor.get('label'),
            'duration': 1.0 * tensor.get('duration')
        }

        return result

    def __len__(self):
        return len(self.list)

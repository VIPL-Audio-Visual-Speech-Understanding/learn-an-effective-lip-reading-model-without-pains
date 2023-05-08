import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from turbojpeg import TurboJPEG, TJPF_GRAY
from load import load_labels

jpeg = TurboJPEG()


class LRWDataset(Dataset):
    def __init__(self, phase: str, args):
        self.labels = load_labels()
        self.list = []
        self.phase = phase
        self.args = args

        if not hasattr(self.args, 'is_aug'):
            setattr(self.args, 'is_aug', True)

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
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop((88, 88)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        else:
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.CenterCrop((88, 88)),
                transforms.ToTensor(),
            ])

        batch_img = torch.stack([transform(img) for img in inputs], dim=0)

        result = {
            'video': batch_img.unsqueeze(1).float(),
            'label': tensor.get('label'),
            'duration': 1.0 * tensor.get('duration')
        }

        return result

    def __len__(self):
        return len(self.list)

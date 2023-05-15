# encoding: utf-8
import os
import cv2
import glob
import time
import torch
import numpy as np
from turbojpeg import TurboJPEG
from torch.utils.data import Dataset, DataLoader

jpeg = TurboJPEG()


def ensure_dir(directory: str) -> None:
    """
    gets a path to directory, if it doesn't exist - create it
    :param directory: path to directory
    :return: None
    """

    if not os.path.exists(directory):
        os.makedirs(directory)


def load_duration(file: str) -> np.array:
    """
    gets a path to an video example file and return its duration calculated by numpy
    :param file: path for video file
    :return: duration of the file
    """

    with open(file, 'r') as f:
        lines = f.readlines()

        for line in lines:
            if line.find('Duration') != -1:
                duration = float(line.split(' ')[1])

    tensor = np.zeros(29)
    mid = 29 / 2
    start = int(mid - duration / 2 * 25)
    end = int(mid + duration / 2 * 25)
    tensor[start:end] = 1.0

    return tensor


def extract_opencv(file_name: str) -> list:
    """
    gets a path to video file, try to extract the ROI from it
    :param file_name: path to video file
    :return: ROI of given video file
    """

    video = []
    cap = cv2.VideoCapture(file_name)

    while cap.isOpened():
        ret, frame = cap.read()  # BGR
        if ret:
            frame = frame[115:211, 79:175]
            frame = jpeg.encode(frame)
            video.append(frame)
        else:
            break
    cap.release()

    return video        


target_dir = 'lrw_roi_80_116_175_211_npy_gray_pkl_jpeg'
ensure_dir(target_dir)


class LRWDataset(Dataset):
    """
    Object that represents the preparation process of LRW dataset.
    Inherits LRWDatasetInterface which implements Dataset of torch module
    and generates training samples of LRW
    """

    def __init__(self):

        with open('label_sorted.txt') as myfile:
            self.labels = myfile.read().splitlines()            
        
        self.list = []

        for i, label in enumerate(self.labels):
            files = glob.glob(os.path.join('lrw_mp4', label, '*', '*.mp4'))

            for file in files:
                file_to_save = file.replace('lrw_mp4', target_dir).replace('.mp4', '.pkl')
                save_path = os.path.split(file_to_save)[0]
                ensure_dir(save_path)

            files = sorted(files)
            self.list += [(file, i) for file in files]
            
    def __getitem__(self, idx: int) -> dict:
        """
        implements the operator []
        :param idx: index to return of the dataset object
        :return: by given index, return the respectively data on that index
        """

        inputs = extract_opencv(self.list[idx][0])

        duration = self.list[idx][0]
        labels = self.list[idx][1]

        result = {'video': inputs,
                  'label': int(labels),
                  'duration': load_duration(duration.replace('.mp4', '.txt')).astype(np.bool)}

        output = self.list[idx][0].replace('lrw_mp4', target_dir).replace('.mp4', '.pkl')
        torch.save(result, output)
        
        return result

    def __len__(self) -> int:
        """
        implements the len operator
        :return: len of self.list
        """

        return len(self.list)


def main():
    loader = DataLoader(LRWDataset(),
                        batch_size=96,
                        num_workers=16,
                        shuffle=False,
                        drop_last=False)

    tic = time.time()

    for i, batch in enumerate(loader):
        toc = time.time()
        eta = ((toc - tic) / (i + 1) * (len(loader) - i)) / 3600.0
        print(f'eta:{eta:.5f}')


if __name__ == '__main__':
    main()

# encoding: utf-8
import numpy as np
import glob
import time
import cv2
import os
from torch.utils.data import Dataset, DataLoader
import torch


def extract_opencv(filename):
    video = []
    cap = cv2.VideoCapture(filename)
    while(cap.isOpened()):
        ret, frame = cap.read() # BGR
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)                    
            video.append(frame)
        else:
            break
    cap.release()
    video = np.array(video)
    video = video[:, 115:211, 79:175]
    return video


target_dir = 'lrw_roi_80_116_175_211_npy_gray_pkl'

if(not os.path.exists(target_dir)):
    os.makedirs(target_dir)    

class LRWDataset(Dataset):
    def __init__(self):

        with open('label_sorted.txt') as myfile:
            self.labels = myfile.read().splitlines()            
        
        self.list = []

        for (i, label) in enumerate(self.labels):
            files = glob.glob(os.path.join('lrw_mp4', label, '*', '*.mp4'))
            for file in files:
                savefile = file.replace('lrw_mp4', target_dir).replace('.mp4', '.pkl')
                savepath = os.path.split(savefile)[0]
                if(not os.path.exists(savepath)):
                    os.makedirs(savepath)
                
            files = sorted(files)
            

            self.list += [(file, i) for file in files]                                                                                
            
        
    def __getitem__(self, idx):
            
        inputs = extract_opencv(self.list[idx][0]).astype(np.uint8)
        result = {}        
         
        name = self.list[idx][0]
        duration = self.list[idx][0]            
        labels = self.list[idx][1]

                    
        result['video'] = inputs
        result['label'] = int(labels)
        result['duration'] = self.load_duration(duration.replace('.mp4', '.txt')).astype(np.bool)
        savename = self.list[idx][0].replace('lrw_mp4', target_dir).replace('.mp4', '.pkl')
        torch.save(result, savename)
        
        return result

    def __len__(self):
        return len(self.list)

    def load_duration(self, file):
        with open(file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if(line.find('Duration') != -1):
                    duration = float(line.split(' ')[1])
        
        tensor = np.zeros(29)
        mid = 29 / 2
        start = int(mid - duration / 2 * 25)
        end = int(mid + duration / 2 * 25)
        tensor[start:end] = 1.0
        return tensor            

if(__name__ == '__main__'):
    loader = DataLoader(LRWDataset(),
            batch_size = 96, 
            num_workers = 8,   
            shuffle = False,         
            drop_last = False)
    
    import time
    tic = time.time()
    for i, batch in enumerate(loader):
        toc = time.time()
        eta = ((toc - tic) / (i + 1) * (len(loader) - i)) / 3600.0
        print(f'eta:{eta:.5f}')        
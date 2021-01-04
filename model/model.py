from .video_cnn import VideoCNN
import torch
import torch.nn as nn
import random
from torch.cuda.amp import autocast, GradScaler


class VideoModel(nn.Module):

    def __init__(self, args, dropout=0.5):
        super(VideoModel, self).__init__()   
        
        self.args = args
        
        self.video_cnn = VideoCNN(se=self.args.se)        
        if(self.args.border):
            in_dim = 512 + 1
        else:
            in_dim = 512
        self.gru = nn.GRU(in_dim, 1024, 3, batch_first=True, bidirectional=True, dropout=0.2)        
            

        self.v_cls = nn.Linear(1024*2, self.args.n_class)     
        self.dropout = nn.Dropout(p=dropout)        

    def forward(self, v, border=None):
        self.gru.flatten_parameters()
        
        if(self.training):                            
            with autocast():
                f_v = self.video_cnn(v)  
                f_v = self.dropout(f_v)        
            f_v = f_v.float()
        else:                            
            f_v = self.video_cnn(v)  
            f_v = self.dropout(f_v)        
        
        if(self.args.border):
            border = border[:,:,None]
            h, _ = self.gru(torch.cat([f_v, border], -1))
        else:            
            h, _ = self.gru(f_v)
        
                                                                                                        
        y_v = self.v_cls(self.dropout(h)).mean(1)
            
        return y_v
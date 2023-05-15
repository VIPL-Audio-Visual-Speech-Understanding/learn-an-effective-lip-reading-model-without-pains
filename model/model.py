from .video_cnn import VideoCNN
import torch
import torch.nn as nn
import random
from torch.cuda.amp import autocast, GradScaler


class Singleton(type):
    """
    Singleton implementation to avoid multiple instantiation of objects
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        else:
            cls._instances[cls].__init__(*args, **kwargs)

        return cls._instances[cls]


class VideoModel(nn.Module, metaclass=Singleton):
    """
    The VideoModel class is a PyTorch nn.Module that consists of a video CNN and a GRU network.
    The VideoModel class takes as input a batch of videos,
    and outputs a tensor of shape (batch_size, n_class) containing the logits for each video in the batch.
    """

    def __init__(self, num_classes: int, dropout: float = 0.5, training: bool = False) -> None:
        super(VideoModel, self).__init__()

        self.num_classes = num_classes
        self.video_cnn = VideoCNN()
        in_dim = 512
        self.gru = nn.GRU(in_dim, 1024, 3, batch_first=True, bidirectional=True, dropout=0.2)
        self.v_cls = nn.Linear(1024 * 2, self.num_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, v: torch.Size) -> torch.Size:
        """
        The output of the CNN is fed into a bidirectional GRU network with 1024 hidden units per direction, and 3 layers.
        The GRU network outputs a tensor of shape (batch_size, seq_len, 2048).
        The output of the GRU network is then fed into a linear layer with n_class output units,
         which produces a tensor of shape (batch_size, n_class) containing the logits for each video in the batch.
        """

        self.gru.flatten_parameters()

        if self.training:
            with autocast():
                f_v = self.video_cnn(v)
                f_v = self.dropout(f_v)
            f_v = f_v.float()
        else:
            f_v = self.video_cnn(v)
            f_v = self.dropout(f_v)

        h, _ = self.gru(f_v)

        y_v = self.v_cls(self.dropout(h)).mean(1)

        return y_v

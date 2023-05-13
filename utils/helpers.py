import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from torch.cuda.amp import autocast
import torch
import matplotlib.pyplot as plt


def parallel_model(model):
    return nn.DataParallel(model)


def load_missing(model, pretrained_dict):
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                       k in model_dict.keys() and v.size() == model_dict[k].size()}
    missed_params = [k for k, v in model_dict.items() if not k in pretrained_dict.keys()]

    print('loaded params/tot params:{}/{}'.format(len(pretrained_dict), len(model_dict)))
    print('miss matched params:', missed_params)
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model


def show_lr(optimizer):
    return ','.join(['{:.6f}'.format(param_group['lr']) for param_group in optimizer.param_groups])


def dataset2dataloader(dataset, batch_size, num_workers, shuffle=True):
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        shuffle=shuffle,
                        drop_last=False,
                        pin_memory=True)
    return loader


def calculate_loss(mixup, alpha, video_model, video, label):
    loss = {}
    loss_fn = nn.CrossEntropyLoss()
    with autocast():
        if mixup:
            mixup_coef = np.random.beta(alpha, alpha)
            shuffled_indices = torch.randperm(video.size(0)).cuda(non_blocking=True)
            mixed_video = mixup_coef * video + (1 - mixup_coef) * video[shuffled_indices, :]
            mixed_label_a, mixed_label_b = label, label[shuffled_indices]
            predicted_label = video_model(mixed_video)
            loss_bp = mixup_coef * loss_fn(predicted_label, mixed_label_a) + (1 - mixup_coef) * loss_fn(predicted_label, mixed_label_b)
        else:
            predicted_label = video_model(video)
            loss_bp = loss_fn(predicted_label, label)
    loss['CE V'] = loss_bp
    return loss


def prepare_data(sample):
    video = sample['video'].cuda(non_blocking=True)
    label = sample['label'].cuda(non_blocking=True).long()
    return video, label


def plot_train_loss(train_losses, epoch):
    print_interval = 5
    if epoch % print_interval == 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(train_losses, label='Training Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss vs. Epoch')
        ax.legend()
        plt.show()


def add_msg(msg, k, v):
    if msg:
        msg += ','
    msg += k.format(v)
    return msg

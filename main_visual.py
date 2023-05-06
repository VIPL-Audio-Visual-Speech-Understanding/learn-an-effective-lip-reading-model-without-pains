import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import numpy as np
import time
from model import VideoModel
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

torch.backends.cudnn.benchmark = True
parser = argparse.ArgumentParser()


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


parser.add_argument('--gpus', type=str, required=True)
parser.add_argument('--lr', type=float, required=True)
parser.add_argument('--batch_size', type=int, required=True)
parser.add_argument('--n_class', type=int, required=True)
parser.add_argument('--num_workers', type=int, required=True)
parser.add_argument('--max_epoch', type=int, required=True)
parser.add_argument('--test', type=str2bool, required=True)

# load opts
parser.add_argument('--weights', type=str, required=False, default=None)

# save prefix
parser.add_argument('--save_prefix', type=str, required=True)

# dataset
parser.add_argument('--dataset', type=str, required=True)
# parser.add_argument('--border', type=str2bool, required=True)
# parser.add_argument('--mixup', type=str2bool, required=True)
# parser.add_argument('--label_smooth', type=str2bool, required=True)
# parser.add_argument('--se', type=str2bool, required=True)

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

if args.dataset == 'lrw':
    from utils import LRWDataset as Dataset
# elif (args.dataset == 'lrw1000'):
#     from utils import LRW1000_Dataset as Dataset
else:
    raise Exception('lrw is supported only')

video_model = VideoModel(args).cuda()

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


lr = args.batch_size / 32.0 / torch.cuda.device_count() * args.lr
optim_video = optim.Adam(video_model.parameters(), lr=lr, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optim_video, T_max=args.max_epoch, eta_min=5e-6)

if args.weights is not None:
    print('load weights')
    weight = torch.load(args.weights, map_location=torch.device('cpu'))
    load_missing(video_model, weight.get('video_model'))

video_model = parallel_model(video_model)


def dataset2dataloader(dataset, batch_size, num_workers, shuffle=True):
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        shuffle=shuffle,
                        drop_last=False,
                        pin_memory=True)
    return loader


def add_msg(msg, k, v):
    if msg != '':
        msg = msg + ','
    msg = msg + k.format(v)
    return msg


def test():
    with torch.no_grad():
        dataset = Dataset('val', args)
        print('Start Testing, Data Length:', len(dataset))
        loader = dataset2dataloader(dataset, args.batch_size, args.num_workers, shuffle=False)

        print('start testing')
        v_acc = []
        total = 0

        for (i_iter, input) in enumerate(loader):

            video_model.eval()

            tic = time.time()
            video = input.get('video').cuda(non_blocking=True)
            label = input.get('label').cuda(non_blocking=True)
            total = total + video.size(0)

            with autocast():
                y_v = video_model(video)

            v_acc.extend((y_v.argmax(-1) == label).cpu().numpy().tolist())
            toc = time.time()
            if i_iter % 10 == 0:
                msg = ''
                msg = add_msg(msg, 'v_acc={:.5f}', np.array(v_acc).reshape(-1).mean())
                msg = add_msg(msg, 'eta={:.5f}', (toc - tic) * (len(loader) - i_iter) / 3600.0)

                print(msg)

        acc = float(np.array(v_acc).reshape(-1).mean())
        msg = 'v_acc_{:.5f}_'.format(acc)

        return acc, msg


def show_lr(optimizer):
    return ','.join(['{:.6f}'.format(param_group['lr']) for param_group in optimizer.param_groups])


def train():
    dataset = Dataset('train', args)
    print('Start Training, Data Length:', len(dataset))

    loader = dataset2dataloader(dataset, args.batch_size, args.num_workers)

    max_epoch = args.max_epoch
    tot_iter = 0
    best_acc = 0.0
    scaler = GradScaler()
    for epoch in range(max_epoch):

        for (i_iter, input) in enumerate(loader):
            tic = time.time()

            video_model.train()
            video = input.get('video').cuda(non_blocking=True)
            label = input.get('label').cuda(non_blocking=True).long()

            loss = {}

            loss_fn = nn.CrossEntropyLoss()

            with autocast():
                    y_v = video_model(video)

                    loss_bp = loss_fn(y_v, label)

            loss['CE V'] = loss_bp

            optim_video.zero_grad()
            scaler.scale(loss_bp).backward()
            scaler.step(optim_video)
            scaler.update()

            toc = time.time()

            msg = 'epoch={},train_iter={},eta={:.5f}'.format(epoch, tot_iter,
                                                             (toc - tic) * (len(loader) - i_iter) / 3600.0)
            for k, v in loss.items():
                msg += ',{}={:.5f}'.format(k, v)
            msg = f",lr={show_lr(optim_video)}"
            msg = msg + str(',best_acc={:2f}'.format(best_acc))
            print(msg)

            if i_iter == len(loader) - 1 or (epoch == 0 and i_iter == 0):

                acc, msg = test()

                if acc > best_acc:
                    saved_file = '{}_iter_{}_epoch_{}_{}.pt'.format(args.save_prefix, tot_iter, epoch, msg)

                    temp = os.path.split(saved_file)[0]
                    if not os.path.exists(temp):
                        os.makedirs(temp)
                    torch.save(
                        {
                            'video_model': video_model.module.state_dict(),
                        }, saved_file)

                if tot_iter != 0:
                    best_acc = max(acc, best_acc)

            tot_iter += 1

        scheduler.step()


if __name__ == '__main__':
    if args.test:
        acc, msg = test()
        print(f'acc={acc}')
        exit()
    train()

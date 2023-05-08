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
from utils.dataset import LRWDataset as Dataset

torch.backends.cudnn.benchmark = True


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    def str2bool(v: str) -> bool:
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

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    return args


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


def add_msg(msg, k, v):
    if msg:
        msg += ','
    msg += k.format(v)
    return msg


@torch.no_grad()
def test():
    dataset = Dataset('val', args)
    print('Start Testing, Data Length:', len(dataset))
    loader = dataset2dataloader(dataset, args.batch_size, args.num_workers, shuffle=False)

    print('start testing')
    v_acc = []
    total = 0

    for i_iter, input in enumerate(loader):
        video_model.eval()

        tic = time.time()
        video = input['video'].cuda(non_blocking=True)
        label = input['label'].cuda(non_blocking=True)

        with autocast():
            y_v = video_model(video)

        v_acc.extend((y_v.argmax(-1) == label).cpu().numpy().tolist())
        toc = time.time()

        if i_iter % 10 == 0:
            msg = add_msg('', 'v_acc={:.5f}', np.array(v_acc).mean())
            msg = add_msg(msg, 'eta={:.5f}', (toc - tic) * (len(loader) - i_iter) / 3600.0)

            print(msg)

    acc = float(np.array(v_acc).mean())
    msg = f'v_acc_{acc:.5f}_'

    return acc, msg


def train():
    dataset = Dataset('train', args)
    print('Start Training, Data Length:', len(dataset))

    loader = dataset2dataloader(dataset, args.batch_size, args.num_workers)

    max_epoch = args.max_epoch
    tot_iter = 0
    best_acc = 0.0
    scaler = GradScaler()
    for epoch in range(max_epoch):

        for i_iteration, sample in enumerate(loader):
            tic = time.time()

            video_model.train()
            video = sample['video'].cuda(non_blocking=True)
            label = sample['label'].cuda(non_blocking=True).long()

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

            msg = f'epoch={epoch},train_iter={tot_iter},eta={(toc - tic) * (len(loader) - i_iteration) / 3600.0:.5f}'
            for k, v in loss.items():
                msg += f',{k}={v:.5f}'
            msg += f",lr={show_lr(optim_video)},best_acc={best_acc:2f}"
            print(msg)

            if i_iteration == len(loader) - 1 or (epoch == 0 and i_iteration == 0):
                acc, msg = test()

                if acc > best_acc:
                    saved_file = f'{args.save_prefix}_iter_{tot_iter}_epoch_{epoch}_{msg}.pt'

                    temp = os.path.split(saved_file)[0]
                    if not os.path.exists(temp):
                        os.makedirs(temp)

                    torch.save({
                        'video_model': video_model.module.state_dict(),
                    }, saved_file)

                if tot_iter != 0:
                    best_acc = max(acc, best_acc)

            tot_iter += 1

        scheduler.step()


if __name__ == '__main__':
    args = parse_arguments()
    video_model = VideoModel(args).cuda()

    lr = args.batch_size / 32.0 / torch.cuda.device_count() * args.lr
    optim_video = optim.Adam(video_model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optim_video, T_max=args.max_epoch, eta_min=5e-6)

    if args.weights is not None:
        print('load weights')
        weight = torch.load(args.weights, map_location=torch.device('cpu'))
        load_missing(video_model, weight.get('video_model'))

    video_model = parallel_model(video_model)
    if args.test:
        acc, msg = test()
        print(f'acc={acc}')
        exit()
    train()

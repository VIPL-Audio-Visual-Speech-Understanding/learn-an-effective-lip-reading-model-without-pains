import argparse
import torch
import os
import numpy as np
import time
from model import VideoModel
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from utils.dataset import LRWDataset as Dataset
from utils import helpers
from model.lrw_dataset import LRWDataset

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

    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--n_class', type=int, required=True)
    parser.add_argument('--max_epoch', type=int, required=True)
    parser.add_argument('--test', type=str2bool, required=True)
    parser.add_argument('--num_workers', type=int, required=False, default=1)
    parser.add_argument('--gpus', type=str, required=False, default='0')
    parser.add_argument('--weights', type=str, required=False, default=None)
    parser.add_argument('--save_prefix', type=str, required=True)
    parser.add_argument('--mixup', type=str2bool, required=False, default=False)
    parser.add_argument('--dataset', type=str, required=False, default='lrw')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    return args


@torch.no_grad()
def test(batch_size, num_workers=1):
    dataset = Dataset('val')
    print('Start Testing, Data Length:', len(dataset))
    loader = helpers.dataset2dataloader(dataset, batch_size, num_workers, shuffle=False)

    print('start testing')
    validation_accuracy = []

    for i_iter, sample in enumerate(loader):
        video_model.eval()

        tic = time.time()
        video, label = helpers.prepare_data(sample)

        with autocast():
            y_v = video_model(video)

        validation_accuracy.extend((y_v.argmax(-1) == label).cpu().numpy().tolist())
        toc = time.time()

        if i_iter % 10 == 0:
            msg = helpers.add_msg('', 'v_acc={:.5f}', np.array(validation_accuracy).mean())
            msg = helpers.add_msg(msg, 'eta={:.5f}', (toc - tic) * (len(loader) - i_iter) / 3600.0)

            print(msg)

    accuracy = float(np.array(validation_accuracy).mean())
    accuracy_msg = f'v_acc_{accuracy:.5f}_'

    return accuracy, accuracy_msg


def train():
    dataset = Dataset('train')
    print('Start Training, Old Data Length:', len(dataset))
    print(f"Old data list len {len(dataset.list)}")
    print(f"Old data list first element {dataset.list[1]}")

    dataset = LRWDataset("train", dataset_prefix="")
    print('Start Training, New Data Length:', len(dataset))
    print(f"New data list len {len(dataset.list)}")
    print(f"New data list first element {dataset.list[1]}")

    loader = helpers.dataset2dataloader(dataset, args.batch_size, args.num_workers)

    max_epoch = args.max_epoch
    alpha = 0.2
    tot_iter = 0
    best_acc = 0.0
    train_losses = []
    scaler = GradScaler()
    for epoch in range(max_epoch):
        train_loss = 0.0
        for i_iteration, sample in enumerate(loader):
            tic = time.time()

            video_model.train()
            video, label = helpers.prepare_data(sample)

            loss = helpers.calculate_loss(args.mixup, alpha, video_model, video, label)

            optim_video.zero_grad()
            scaler.scale(loss['CE V']).backward()
            scaler.step(optim_video)
            scaler.update()

            train_loss += loss['CE V'].item()
            toc = time.time()

            msg = f'epoch={epoch},train_iter={tot_iter},eta={(toc - tic) * (len(loader) - i_iteration) / 3600.0:.5f}'
            for k, v in loss.items():
                msg += f',{k}={v:.5f}'
            msg += f",lr={helpers.show_lr(optim_video)},best_acc={best_acc:2f}"
            print(msg)

            if i_iteration == len(loader) - 1 or (epoch == 0 and i_iteration == 0):
                acc, msg = test(args.batch_size)

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

        train_losses.append(train_loss / len(loader))
        helpers.plot_train_loss(train_losses, epoch)

        scheduler.step()


if __name__ == '__main__':
    args = parse_arguments()
    video_model = VideoModel(args.n_class).cuda()

    lr = args.batch_size / 32.0 / torch.cuda.device_count() * args.lr
    optim_video = optim.Adam(video_model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optim_video, T_max=args.max_epoch, eta_min=5e-6)

    if args.weights is not None:
        print('load weights')
        weight = torch.load(args.weights, map_location=torch.device('cpu'))
        helpers.load_missing(video_model, weight.get('video_model'))

    video_model = helpers.parallel_model(video_model)
    if args.test:
        acc, msg = test(args.batch_size)
        print(f'acc={acc}')
        exit()
    train()

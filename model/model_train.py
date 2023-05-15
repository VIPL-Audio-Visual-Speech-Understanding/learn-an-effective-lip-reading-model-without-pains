
def train(scheduler):
    dataset = Dataset('train')
    print('Start Training, Data Length:', len(dataset))

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

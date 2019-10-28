#coding=utf-8

#  import dependency library
import os
import time
import logging
import random
import argparse
import setproctitle
import numpy as np
#  torch library
import torch
import torch.optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

#  user defined library
import models
from utils import ParserUse, criterions
from utils.show_train import Visualizer
from utils.train_utils import adjust_learning_rate
#  datasets
from data import datasets
from data.data_utils import init_fn
from data.sampler import CycleSampler


cudnn.benchmark = True  # let auto-tuner find the most efficient network (used in fixed network.)

# set parameters
parser = argparse.ArgumentParser()

parser.add_argument('-cfg', '--cfg', default='DMFNet_MEMB3D_TRAIN', required=True, type=str, help='Your detailed configuration of the network')
parser.add_argument('-gpu', '--gpu', default='0', type=str, required=True, help='Supprot one GPU & multiple GPUs.')
parser.add_argument('-batch_size', '--batch_size', default=1, type=int, help='Batch size')
parser.add_argument('-restore', '--restore', default='', type=str)
parser.add_argument('--show_image_freq', default=20, type=int, help="frequency of showing image")
parser.add_argument('--show_loss_freq', default=5, type=int, help="frequency of showing loss")

path = os.path.dirname(__file__)

## parse arguments
args = parser.parse_args()
args = ParserUse(args.cfg, log='train').add_args(args)

cpkts = args.makedir()
args.resume = os.path.join(cpkts, args.restore)

if args.show_image_freq > 0 or args.show_loss_freq > 0:
    visualizer = Visualizer(1)

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    #=====================================================
    #   set seed for randomization
    #=====================================================
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    #=====================================================
    #  construct network
    #=====================================================
    Network = getattr(models, args.net)
    model = Network(**args.net_params)
    model = torch.nn.DataParallel(model).cuda()
    optimizer = getattr(torch.optim, args.opt)(model.parameters(), **args.opt_params)
    criterion = getattr(criterions, args.criterion)

    #=====================================================
    #  resume network
    #=====================================================
    msg = ""  # msg to log
    if args.resume:
        if os.path.isfile(args.resume):
            print("==>Loading checking point '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_iter = checkpoint("iter")
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optim_dict"])
            msg = "==>Finish loading checkpoint {}".format(args.resume)
        else:
            msg = "==>!!! Cannot find checkpoint at {}".format(args.resume)
    else:
        msg = "-"*20 + "New training" + "-"*20
    msg += "\n" + str(args)
    logging.info(msg)

    #=====================================================
    #  set dataset loader
    #=====================================================
    Dataset = getattr(datasets, args.dataset)
    train_set = Dataset(root=args.train_data_dir, membrane_names=args.train_embryos, for_train=True, transforms=args.train_transforms, return_target=True)
    num_iters = args.num_iters or (len(train_set) * args.num_epochs) // args.batch_size
    num_iters -= args.start_iter
    train_sampler = CycleSampler(len(train_set), num_iters*args.batch_size)
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        collate_fn=train_set.collate,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=init_fn
    )

    #====================================================
    #  start training
    #====================================================
    start = time.time()
    torch.set_grad_enabled(True)
    enum_batches = len(train_set) / float(args.batch_size)  # number of epoches in each iteration
    for i, data in enumerate(train_loader, args.start_iter):
        #  record process
        elapsed_bsize = int(i / enum_batches) + 1
        epoch = int((i + 1) / enum_batches)
        setproctitle.setproctitle("Epoch: {}/{}".format(elapsed_bsize, args.num_epochs))  # set process name

        #  adjust learning super parameters
        adjust_learning_rate(optimizer, epoch, args.num_epochs, args.opt_params.lr)

        #  go through the network
        data = [t.cuda(non_blocking=True) for t in data]  # Set non_blocking for multiple GPUs
        raw, target, tp = data[:3]
        output = model(raw, tp)

        #  get loss
        if not args.weight_type:
            args.weight_type = "square"
        if args.criterion_kwargs is not None:
            loss = criterion(output, target, **args.criterion_kwargs)
        else:
            loss = criterion(output, target)

        #  backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #=============================================================
        #   Show mmiddle results
        if args.show_image_freq > 0 and (i % args.show_image_freq) == 0:
            image_dict = dict(Raw=raw[0, 0, :, :, 60], Target=target[0, :, :, 60].float(), Prediction=output[0, 1, :, :, 60])
            visualizer.show_current_images(image_dict)
        if args.show_loss_freq > 0 and (i % args.show_loss_freq) == 0:
            visualizer.plot_current_losses(progress_ratio=(i+1)/enum_batches, losses=dict(Diceloss=loss.item()))
        # =============================================================

        #  save trained model
        if (i + 1) % int(enum_batches * args.save_freq) == 0:
            file_name = os.path.join(cpkts, "model_epoch_{}.pth".format(epoch))
            torch.save(dict(
                iter=i,
                state_dict=model.state_dict(),
                optim_dict=optimizer.state_dict()
            ), file_name)

        logging.info("Iter {0:}, Epoch {1:.4f}, Loss {2:.7f}".format(i+1, (i+1)/enum_batches, loss.item()))

    #  save the last model
    i = num_iters + args.start_iter
    file_name = os.path.join(cpkts, "model_last.pth")
    torch.save(dict(
        iter=i,
        state_dict=model.state_dict(),
        optim_dict=optimizer.state_dict()
    ), file_name)


if __name__ == "__main__":
    main()



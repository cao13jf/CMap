#  import dependency library
import os
import argparse
import time
import logging
import random
import shutil
import numpy as np
import torch
import setproctitle
import torch.optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

#  import user's library
import models
from data import datasets
from utils import ParserUse, str2bool
from utils.prediction_utils import validate

cudnn.benchmark = True
path = os.path.dirname(__file__)

parser = argparse.ArgumentParser()
parser.add_argument("--cfg", default="DMFNET_MEMB3D_TEST", type=str, help="Specify experiment parameter file")
parser.add_argument("--trained_model", default="model_last.pth", type=str, help="Specified the trained model")
parser.add_argument("--mode", default=2, type=int, help="0 -- cross-validation on the training set;"
                                                                       "1 -- validing on the validation set"
                                                                       "2 -- testing on the testing set")
parser.add_argument("--gpu", default="0", type=str, help="GPUs used for training")
parser.add_argument("--suffix", default="*.pkl", type=str, help="Suffix used fo filter data")
parser.add_argument("--save_result", default=False, type=str2bool, help="Whether save the result")
parser.add_argument("--save_format", default="nii", type=str, help="Format of saved file")
parser.add_argument("--snap_shot", default=False, type=str2bool, help="Whether save one snapped slice for the result")
args = parser.parse_args()
args = ParserUse(args.cfg, log="test").add_args(args)
args.gpu = str(args.gpu)
ckpts = args.makedir()
args.resume = os.path.join(ckpts, args.trained_model)

#=========================================================
#  main program for prediction
#=========================================================
def main():
    setproctitle.setproctitle(parser.cfg)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    assert torch.cuda.is_available(), "CPU is needed for prediction"


    #=============================================================
    #  set seeds for randomlization
    #=============================================================
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    #=============================================================
    #  construct network model
    #=============================================================
    Network = getattr(models, args.net)
    model = Network(**args.net_params)
    model = torch.nn.DataParallel(model).cuda()
    print("="*20 + "Loading parameters {}".format(args.resume) + "="*20)
    assert os.path.isfile(args.resume), "{} ".format(args.resume) + "doesn't exist"
    check_point = torch.load(args.resume)
    args.start_iter = check_point["iter"]
    model.load_state_dict(check_point["state_dict"])

    msg = ("Loaded checkpoint '{}' (iter {})".format(args.resume, check_point["iter"]))
    masg = "\n" + str(args)
    logging.info(msg)

    #=============================================================
    #  set data loader
    #=============================================================
    if args.mode == 0:
        root_path = args.train_data_dir
        is_scoring = True
    elif args.mode == 1:
        root_path = args.valid_data_dir
        is_scoring = True
    elif args.mode == 2:
        root_path = args.test_data_dir
        is_scoring = False
    else:
        raise ValueError("Choose mode from '0--train, 1--valid, 2--test'")
    if args.save_result:
        if args.test_embryos is None:
            args.test_embryos = [name for name in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, name))]
    Dataset = getattr(datasets, args.dataset)
    test_set = Dataset(root=root_path, membrane_names=args.test_embryos, for_train=False, transforms=args.test_transforms, return_target=is_scoring, suffix=args.suffix)
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=1,
        shuffle=False,
        collate_fn=test_set.collate, # control how data is stacked
        num_workers=10,
        pin_memory=True
    )

    #=============================================================
    #  begin prediction
    #=============================================================
    #  Prepare (or clear) in order to update all files
    if args.save_result:
        for embryo_name in args.test_embryos:
            if os.path.isdir(os.path.join("./output", embryo_name)):
                shutil.rmtree(os.path.join("./output", embryo_name))
    logging.info("-"*50)
    logging.info(msg)

    with torch.no_grad():
        validate(
            valid_loader=test_loader,  # dataset loader
            model=model,  # model
            savepath="./output",  # output folder
            names=test_set.names,  # stack name lists
            scoring=False,  # whether keep accuracy
            save_format=".nii.gz",  # save volume format
            snapsot=False,  # whether keep snap
            postprocess=False,
        )

if __name__ == "__main__":
    main()
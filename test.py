#  import dependency library
import os
import argparse
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
from utils import ParserUse
from data.preprocess import niigz_to_pkl_run
from utils.show_train import Visualizer
from utils.prediction_utils import validate, membrane2cell, combine_dividing_cells
from utils.shape_analysis import shape_analysis_func
from utils.qc import generate_qc
from utils.generate_gui_data import generate_gui_data

cudnn.benchmark = True # https://zhuanlan.zhihu.com/p/73711222 to accelerate the network
path = os.path.dirname(__file__)

parser = argparse.ArgumentParser()
parser.add_argument("--cfg", default="DMFNET_MEMB3D_TEST", type=str, help="Specify experiment parameter file")
parser.add_argument("--mode", default=2, type=int, help="0 -- cross-validation on the training set;"
                                                                       "1 -- validing on the validation set"
                                                                       "2 -- testing on the testing set")
parser.add_argument("--gpu", default="0", type=str, help="GPUs used for training")
parser.add_argument("--ckpts", default="./ckpts", type=str, help="GPUs used for training")
parser.add_argument("--suffix", default="*.pkl", type=str, help="Suffix used fo filter data")
parser.add_argument("--save_format", default="nii", type=str, help="Format of saved file")
args = parser.parse_args()
args = ParserUse(args.cfg, log="test").add_args(args)
args.gpu = str(args.gpu)
ckpts = args.makedir()
args.resume = args.trained_model

# ===========================================
# Snap visualizer
# ===========================================
visualizer = None
if args.show_snap:
    visualizer = Visualizer(1)

# =========================================================
#  main program for prediction
# =========================================================
def main():
    setproctitle.setproctitle(args.cfg)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    assert torch.cuda.is_available(), "CPU is needed for prediction"

    # =============================================================
    #  set seeds for randomlization in TRAIN_TEST.yaml
    # =============================================================
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # start to predict the unconstant eggshell (prediciton, binary segmentation)
    if args.get_memb_bin:
        # get membrane binary shell
        test_folder = dict(root="dataset/test", has_label=False)
        # nii.gz to pickle, make it easier to read in neural network
        niigz_to_pkl_run(test_folder, embryo_names=args.test_embryos, max_times=args.max_times)
        # =============================================================
        #  construct network model
        # =============================================================
        Network = getattr(models, args.net)
        model = Network(**args.net_params)
        model = torch.nn.DataParallel(model).cuda()
        print("="*20 + "Loading parameters {}".format(args.resume) + "="*20)
        assert os.path.isfile(args.resume), "{} ".format(args.resume) + "doesn't exist"
        check_point = torch.load(args.resume)
        args.start_iter = check_point["iter"]
        model.load_state_dict(check_point["state_dict"])

        msg = ("Loaded checkpoint '{}' (iter {})".format(args.resume, check_point["iter"]))
        msg = msg + "\n" + str(args)
        logging.info(msg)

        # =============================================================
        #    set data loader
        # =============================================================
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
        if args.get_memb_bin or args.get_cell:
            if args.test_embryos is None:
                args.test_embryos = [name for name in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, name))]
        Dataset = getattr(datasets, args.dataset)
        test_set = Dataset(root=root_path, membrane_names=args.test_embryos, for_train=False, transforms=args.test_transforms,
                           return_target=is_scoring, suffix=args.suffix, max_times=args.max_times)
        test_loader = DataLoader(
            dataset=test_set,
            batch_size=1,
            shuffle=False,
            # collate_fn=test_set.collate, # control how data is stacked
            num_workers=10,
            pin_memory=True
        )

        #=============================================================
        #  begin prediction
        #=============================================================
        #  Prepare (or clear) in order to update all files
        if args.get_memb_bin or args.get_cell:
            for embryo_name in args.test_embryos:
                if os.path.isdir(os.path.join("./output", embryo_name)):
                    shutil.rmtree(os.path.join("./output", embryo_name))
        logging.info("-"*50)
        logging.info(msg)

        # the file will save in the segMemb folder
        with torch.no_grad():
            validate(
                valid_loader=test_loader,  # dataset loader
                model=model,  # model
                savepath= "./output",  # output folder
                names=test_set.names,  # stack name lists
                scoring=False,  # whether keep accuracy
                save_format=".nii.gz",  # save volume format
                snapsot=visualizer,  # whether keep snap
                postprocess=False,
                size=test_set.size
            )


    #  Post process on binary segmentation. Group them into closed 3D cells
    if args.get_cell:
        # read the binary segmentation in segMemb folder and process them
        membrane2cell(args)

    #  Combine labels based on dividing cells
    if args.tp_combine:
        print("Begin combine division based on TP...\n")
        combine_dividing_cells(args)

    if args.shape_analysis:
        print("Begin collect cell shape information to do cell shape statistics...\n")
        shape_analysis_func(args)

    if args.get_volume_var:
        # --------QC, children volume difference, volume difference to previous TP, volume divergence of cell
        print("Begin collect variations of volume and surface")
        generate_qc(args)

    if args.gui_data:
        print("Begin generate GUI data")
        generate_gui_data(args)

if __name__ == "__main__":
    main()
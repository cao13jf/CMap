#coding=utf-8

#  import dependency library
import os
import time
import argparse
import numpy as np
#  torch library
import torch
import torch.optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

#  datasets
from data import datasets
from data.sampler import CycleSampler

#  user defined library
import models
from utils import ParserUse, criterions


cudnn.benchmark = True  # let auto-tuner find the most efficient network (used in fixed network.)

# set parameters
parser = argparse.ArgumentParser()
parser.add_argument("-cfg", "--cfg", default="MEMBRANE3D_TRAIN", required=True, type=str, help="training setting files")
parser.add_argument("-gpu", "--gpu", default="0", type=str, required=True, help="GPUs")
parser.add_argument("-batch_size", "--batch_size", default=1, type=int, required=True, help="training batch size")
parser.add_argument("-restore", "--restore", default="model_last.pth", type=str)
path = os.path.dirname(__file__)
args = parser.parse_args()
args = ParserUse(args.cfg, log="train").add_cfg()
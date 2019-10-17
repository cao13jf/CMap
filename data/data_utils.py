# import denpendency library
import os
import glob
import pickle
import torch
import random
import numpy as np

M = 2**32 -1

def get_all_stack(root, membrane_list, suffix):
    file_list = []
    for membrane in membrane_list:
        stacks = glob.glob(os.path.join(root, membrane, "PklFile", suffix))
        file_list = file_list + stacks
    return file_list

def pkload(fname):
    with open(fname, "rb") as f:
        return pickle.load(f)

#  initilization function for workers
def init_fn(worker):
    seed = torch.LongTensor(1).random_().item()
    seed = (seed + worker) % M
    np.random.seed(seed)
    random.seed(seed)

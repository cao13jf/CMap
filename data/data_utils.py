# import denpendency library
import os
import glob
import pickle


def get_all_stack(root, membrane_list, suffix):
    file_list = []
    for membrane in membrane_list:
        stacks = glob.glob(os.path.join(root, membrane, suffix))
        file_list = file_list + stacks
    return file_list

def pkload(fname):
    with open(fname, "rb") as f:
        return pickle.load(f)



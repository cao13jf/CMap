'''
Preprocess for fast data reading
'''
#  Import dependency library
import os
import glob
import nibabel as nib

#  import user defined library
from utils.data_io import nib_load, normalize3d, pkl_save
from utils import ParserUse

args = ParserUse()

# dataset folder
train_folder = dict(root="dataset/train", has_label=True)
test_folder = dict(root="dataset/test", has_label=False)

def nii_to_pkl(path, has_label=True):
    pkl_folder = os.path.join()

def doit(target, embryo_names=None):
    #  get embryo list
    if embryo_names is None:

    root, has_label = target["root"], target["has_label"]
    nii_to_pkl(root, has_label)

if __name__ == "__main__":
    embryo_names = None
    doit(train_folder, embryo_names)
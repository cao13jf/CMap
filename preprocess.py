'''
Preprocess for fast data reading
'''
#  Import dependency library
import os
import glob
from tqdm import tqdm
import nibabel as nib

#  import user defined library
from utils.data_io import nib_load, normalize3d, pkl_save
from utils import ParserUse

args = ParserUse()

# dataset folder
train_folder = dict(root="dataset/train", has_label=True)
test_folder = dict(root="dataset/test", has_label=False)

def nii_to_pkl(embryo_path, has_label=True):
    #  build pkl folder
    pkl_folder = os.path.join(embryo_path, "PklFile")
    if not os.path.isdir(pkl_folder):
        os.mkdir(pkl_folder)
    #  get data list
    raw_memb_list = glob.glob(os.path.join(embryo_path, "RawMemb", "*.gz"))
    raw_nuc_list = glob.glob(os.path.join(embryo_path, "RawNuc", "*.gz"))
    seg_nuc_list = glob.glob(os.path.join(embryo_path, "SegNuc", "*.gz"))
    if has_label:
        seg_memb_list = glob.glob(os.path.join(embryo_path, "SegMemb", "*.gz"))
        seg_cell_list = glob.glob(os.path.join(embryo_path, "SegCell", "*.gz"))
    #  read nii and save data as pkl
    for i, raw_memb_file in tqdm(enumerate(raw_memb_list), desc="saving"+embryo_path):
        base_name = os.path.basename(raw_memb_file).split(".")[0].split("_")
        base_name = base_name[0] + "_" + base_name[1]
        raw_memb = nib_load(raw_memb_file)  # TODO: how to normalize
        raw_nuc = nib_load(raw_nuc_list[i])
        seg_nuc = nib_load(seg_nuc_list[i])
        if has_label:
            seg_memb = nib_load(seg_memb_list[i])
            seg_cell = nib_load(seg_cell_list[i])

        pickle_file = os.path.join(pkl_folder, base_name+'.pkl')
        if has_label:
            pkl_save([raw_memb, raw_nuc, seg_nuc, seg_memb, seg_cell], pickle_file)
        else:
            pkl_save([raw_memb, raw_nuc, seg_nuc], pickle_file)

def doit(target, embryo_names=None):
    #  get embryo list
    root, has_label = target["root"], target["has_label"]
    if embryo_names is None:
        embryo_names = [name for name in os.listdir(root) if os.listdir(os.path.join(root, name))]
    for embryo_name in embryo_names:
        nii_to_pkl(os.path.join(root, embryo_name), has_label)

if __name__ == "__main__":
    embryo_names = None
    doit(train_folder, embryo_names)
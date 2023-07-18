'''
Preprocess for fast data reading
'''
#  Import dependency library
import os
import glob
from tqdm import tqdm

#  import user defined library
from utils.data_io import nib_load, normalize3d, pkl_save


def nii_to_pkl(embryo_path, has_membrane_label=True, has_nucleus_label=True, max_time=None):
    #  build pkl folder
    pkl_folder = os.path.join(embryo_path, "PklFile")
    if not os.path.isdir(pkl_folder):
        os.makedirs(pkl_folder)
    #  get data list
    raw_memb_list = sorted(glob.glob(os.path.join(embryo_path, "RawMemb", "*.gz")))[:max_time]
    raw_nuc_list = sorted(glob.glob(os.path.join(embryo_path, "RawNuc", "*.gz")))[:max_time]
    if has_nucleus_label:
        seg_nuc_list = sorted(glob.glob(os.path.join(embryo_path, "SegNuc", "*.gz")))[:max_time]
    if has_membrane_label:
        seg_memb_list = glob.glob(os.path.join(embryo_path, "SegMemb", "*.gz"))
        seg_cell_list = glob.glob(os.path.join(embryo_path, "SegCell", "*.gz"))
    #  read nii and save data as pkl
    for i, raw_memb_file in enumerate(tqdm(raw_memb_list, desc="saving" + embryo_path)):
        base_name = os.path.basename(raw_memb_file).split("_")
        base_name = base_name[0] + "_" + base_name[1]
        seg_nuc = nib_load(seg_nuc_list[i]) if len(raw_nuc_list) > 0 else None
        raw_nuc = nib_load(raw_nuc_list[i]) if len(raw_nuc_list) > 0 else None
        raw_memb = nib_load(raw_memb_file)  #
        if has_membrane_label:
            seg_memb = nib_load(seg_memb_list[i])
            seg_cell = nib_load(seg_cell_list[i])


        pickle_file = os.path.join(pkl_folder, base_name + '.pkl')
        if has_membrane_label and has_nucleus_label:
            pkl_save(dict(raw_memb=raw_memb, raw_nuc=raw_nuc, seg_nuc=seg_nuc, seg_memb=seg_memb, seg_cell=seg_cell),
                     pickle_file)
        elif has_nucleus_label:
            pkl_save(dict(raw_memb=raw_memb, raw_nuc=raw_nuc, seg_nuc=seg_nuc), pickle_file)
        else:
            pkl_save(dict(raw_memb=raw_memb, raw_nuc=raw_nuc), pickle_file)


def niigz_to_pkl_run(target, embryo_names=None, max_times=None):
    #  get embryo list
    root, has_membrane_label, has_nucleus_label = target["root"], target["has_membrane_label"], target[
        'has_nucleus_label']
    if embryo_names is None:
        embryo_names = [name for name in os.listdir(root) if os.listdir(os.path.join(root, name))]
    for i_embryo, embryo_name in enumerate(embryo_names):
        nii_to_pkl(os.path.join(root, embryo_name), has_membrane_label, has_nucleus_label,max_times[i_embryo])


if __name__ == "__main__":
    # embryo_names = ["191108plc1p1", "200109plc1p1", "200113plc1p2", "200113plc1p3", "200322plc1p2", "200323plc1p1", "200326plc1p3", "200326plc1p4"]
    # max_times = [205, 205, 255, 195, 195, 185, 220, 195]
    embryo_names = ["200117plc1pop1ip2", "200117plc1pop1ip3"]
    max_times = [140, 155]
    # doit(train_folder, embryo_names)
    # dataset folder
    # train_folder = dict(root="dataset/train", has_label=True)
    test_folder = dict(root="dataset/test", has_label=False)
    niigz_to_pkl_run(test_folder, embryo_names, max_times=max_times)

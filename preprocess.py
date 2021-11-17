import os
import glob
import pickle
import warnings
import shutil
warnings.filterwarnings("ignore")
import numpy as np
from PIL import Image
import nibabel as nib
import pandas as pd
from skimage.exposure import rescale_intensity

import pandas
from tqdm import tqdm
import multiprocessing as mp
from skimage.transform import resize

from utils.data_structure import read_new_cd


def combine_slices(config):
    """
    Combine slices into stack images
    :param config: parameters
    :return:
    """
    # signal.emit(True,'sss')
    num_slice = config["num_slice"]
    embryo_names = config["embryo_names"]
    max_time = config["max_time"]
    xy_res = config["xy_resolution"]
    z_res = config["z_resolution"]
    out_size = config["out_size"]
    raw_folder = config["raw_folder"]
    stack_folder = config["target_folder"]
    save_nuc = config["save_nuc"]
    save_memb = config["save_memb"]
    number_dictionary = config["number_dictionary"]

    # get output size
    raw_memb_files = glob.glob(os.path.join(raw_folder, embryo_names[0], "tifR", "*.tif"))
    raw_size = list(np.asarray(Image.open(raw_memb_files[0])).shape) + [int(num_slice * z_res / xy_res)]
    out_res = [res * x / y for res, x, y in zip([xy_res, xy_res, xy_res], raw_size, out_size)]

    # multiprocessing
    mpPool = mp.Pool(mp.cpu_count() - 1)

    for embryo_name in embryo_names:

        # get lineage file
        if config["lineage_file"]:
            lineage_file = os.path.join(config["raw_folder"], embryo_name, "aceNuc", "CD{}.csv".format(embryo_name))
        else:
            lineage_file = None

        # ======================= || save nucleus
        if save_nuc:
            origin_folder = os.path.join(raw_folder, embryo_name, "tif")
            target_folder = os.path.join(stack_folder, embryo_name, "RawNuc")
            if not os.path.isdir(target_folder):
                os.makedirs(target_folder)

            configs = []
            for tp in range(1, max_time + 1):
                configs.append((origin_folder, target_folder, embryo_name, tp, out_size, num_slice, out_res))

            for idx, _ in enumerate(tqdm(mpPool.imap_unordered(stack_nuc_slices, configs), total=len(configs),
                                         desc="1/3 Stack nucleus of {}".format(embryo_name))):
                pass

        # =============================
        if save_memb:
            origin_folder = os.path.join(raw_folder, embryo_name, "tifR")
            target_folder = os.path.join(stack_folder, embryo_name, "RawMemb")
            if not os.path.isdir(target_folder):
                os.makedirs(target_folder)

            configs = []
            for tp in range(1, max_time + 1):
                configs.append((origin_folder, target_folder, embryo_name, tp, out_size, num_slice, out_res))
            for idx, _ in enumerate(tqdm(mpPool.imap_unordered(stack_memb_slices, configs), total=len(configs),
                                         desc="2/3 Stack membrane of {}".format(embryo_name))):
                # TODO: Process Name: `2/3 Stack membrane`; Current status: `idx`; Final status: max_time
                pass

        # save nucleus
        if lineage_file:
            target_folder = os.path.join(stack_folder, embryo_name, "SegNuc")
            if not os.path.isdir(target_folder):
                os.makedirs(target_folder)
            pd_lineage = read_new_cd(lineage_file)

            pd_number = pd.read_csv(number_dictionary, names=["name", "label"])
            number_dict = pd.Series(pd_number.label.values, index=pd_number.name).to_dict()

            configs = []
            for tp in range(1, max_time + 1):
                configs.append((embryo_name, number_dict, pd_lineage, tp, raw_size, out_size, out_res,
                                xy_res / z_res, target_folder))
                # save_nuc_seg(configs[0])
            for idx, _ in enumerate(tqdm(mpPool.imap_unordered(save_nuc_seg, configs), total=len(configs),
                                         desc="3/3 Construct nucleus location of {}".format(embryo_name))):
                # TODO: Process Name: `3/3 Construct nucleus location`; Current status: `idx`; Final status: max_time
                pass
            # for tp in range(1, max_time+1):
            #     save_nuc_seg(embryo_name=embryo_name,
            #                  name_dict=name_dict,
            #                  pd_lineage=pd_lineage,
            #                  tp=tp,
            #                  raw_size=raw_size,
            #                  out_size=out_size,
            #                  out_res=out_res,
            #                  dif_res=xy_res/z_res,
            #                  save_folder=target_folder)
            shutil.copy(lineage_file, os.path.join(stack_folder, embryo_name))

# ============================================
# save raw nucleus stack
# ============================================
def stack_nuc_slices(para):
    [raw_folder, save_folder, embryo_name, tp, out_size, num_slice, res] = para

    out_stack = []
    save_file_name = "{}_{}_rawNuc.nii.gz".format(embryo_name, str(tp).zfill(3))
    for i_slice in range(1, num_slice + 1):
        raw_file_name = "{}_L1-t{}-p{}.tif".format(embryo_name, str(tp).zfill(3), str(i_slice).zfill(2))

        img = np.asanyarray(Image.open(os.path.join(raw_folder, raw_file_name)))
        out_stack.insert(0, img)
    img_stack = np.transpose(np.stack(out_stack), axes=(1, 2, 0))
    img_stack = resize(image=img_stack, output_shape=out_size, preserve_range=True, order=1).astype(np.uint8)
    nib_stack = nib.Nifti1Image(img_stack, np.eye(4))
    nib_stack.header.set_xyzt_units(xyz=3, t=8)
    nib_stack.header["pixdim"] = [1.0, res[0], res[1], res[2], 0., 0., 0., 0.]
    nib.save(nib_stack, os.path.join(save_folder, save_file_name))

# ============================================
# save raw membrane stack
# ============================================
def stack_memb_slices(para):
    [raw_folder, save_folder, embryo_name, tp, out_size, num_slice, res] = para

    out_stack = []
    save_file_name = "{}_{}_rawMemb.nii.gz".format(embryo_name, str(tp).zfill(3))
    for i_slice in range(1, num_slice + 1):
        raw_file_name = "{}_L1-t{}-p{}.tif".format(embryo_name, str(tp).zfill(3), str(i_slice).zfill(2))

        img = np.asanyarray(Image.open(os.path.join(raw_folder, raw_file_name)))
        out_stack.insert(0, img)
    img_stack = np.transpose(np.stack(out_stack), axes=(1, 2, 0))
    # v_min, v_max = np.percentile(img_stack, (0.2, 99.9))
    # img_stack = rescale_intensity(img_stack, in_range=(v_min, v_max))
    img_stack = resize(image=img_stack, output_shape=out_size, preserve_range=True, order=1).astype(np.uint8)
    nib_stack = nib.Nifti1Image(img_stack, np.eye(4))
    nib_stack.header.set_xyzt_units(xyz=3, t=8)
    nib_stack.header["pixdim"] = [1.0, res[0], res[1], res[2], 0., 0., 0., 0.]
    nib.save(nib_stack, os.path.join(save_folder, save_file_name))

# =============================================
# save nucleus segmentation
# =============================================
def save_nuc_seg(para):
    [embryo_name, name_dict, pd_lineage, tp, raw_size, out_size, out_res, dif_res, save_folder] = para

    zoom_ratio = [y / x for x, y in zip(raw_size, out_size)]
    tp_lineage = pd_lineage[pd_lineage["time"] == tp]
    tp_lineage.loc[:, "x"] = (tp_lineage["x"] * zoom_ratio[0]).astype(np.int16)
    tp_lineage.loc[:, "y"] = (np.floor(tp_lineage["y"] * zoom_ratio[1])).astype(np.int16)
    tp_lineage.loc[:, "z"] = (out_size[2] - np.floor(tp_lineage["z"] * (zoom_ratio[2] / dif_res))).astype(np.int16)

    # !!!! x <--> y
    nuc_dict = dict(
        zip(tp_lineage["cell"], zip(tp_lineage["y"].values, tp_lineage["x"].values, tp_lineage["z"].values)))
    labels = [name_dict[name] for name in list(nuc_dict.keys())]
    locs = list(nuc_dict.values())
    out_seg = np.zeros(out_size, dtype=np.uint16)
    out_seg[tuple(zip(*locs))] = labels

    save_file_name = "_".join([embryo_name, str(tp).zfill(3), "segNuc.nii.gz"])
    nib_stack = nib.Nifti1Image(out_seg, np.eye(4))
    nib_stack.header.set_xyzt_units(xyz=3, t=8)
    nib_stack.header["pixdim"] = [1.0, out_res[1], out_res[0], out_res[2], 0., 0., 0., 0.]
    nib.save(nib_stack, os.path.join(save_folder, save_file_name))


if __name__ == "__main__":

    config = dict(num_slice=92,
                  embryo_names=["200117plc1pop1ip3"],
                  max_time = 155,
                  xy_resolution = 0.09,
                  z_resolution = 0.42,
                  out_size = [256, 356, 214],
                  raw_folder = r"F:\ProjectData\MembraneProject\AllRawData",
                  target_folder = r"D:\ProjectData\AllDataPacked",
                  save_nuc=True,
                  save_memb=True,
                  lineage_file = True,
                  number_dictionary = r"D:\OneDriveBackup\OneDrive - City University of Hong Kong\paper\7_AtlasCell\DatasetUpdated\number_dictionary.csv")

    combine_slices(config)
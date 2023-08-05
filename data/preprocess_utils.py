import nibabel as nib
import numpy as np
from PIL import Image
from skimage.transform import resize
from skimage.exposure import rescale_intensity

import os

from .data_utils import check_folder
from .data_structure import read_cd_file

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
    save_file = os.path.join(save_folder, save_file_name)
    check_folder(save_file)
    nib.save(nib_stack, save_file)

# ============================================
# save raw membrane stack
# ============================================
def stack_memb_slices(para):
    [raw_folder, save_folder, embryo_name, tp, out_size, num_slice, res] = para

    out_stack = []
    save_file_name = "{}_{}_rawMemb.nii.gz".format(embryo_name, str(tp).zfill(3))
    for i_slice in range(1, num_slice+1):
        # todo: change your tif file name format here to gurantee the right tif image reading
        raw_file_name = "{}_L1-t{}-p{}.tif".format(embryo_name, str(tp).zfill(3), str(i_slice).zfill(2))
        # transform the image to array and short them in a list
        img = np.asanyarray(Image.open(os.path.join(raw_folder, raw_file_name)))
        out_stack.insert(0, img)
    img_stack = np.transpose(np.stack(out_stack), axes=(1, 2, 0)) # trasnpose the image from zxy to xyz
    v_min, v_max = np.percentile(img_stack, (0.2, 99.9)) # erase the outrange grayscale
    img_stack = rescale_intensity(img_stack, in_range=(v_min, v_max), out_range=(0, 255.0))
    # cut xy, interpolate z
    img_stack = resize(image=img_stack, output_shape=out_size, preserve_range=True, order=1).astype(np.uint8)
    nib_stack = nib.Nifti1Image(img_stack, np.eye(4))
    nib_stack.header.set_xyzt_units(xyz=3, t=8)
    nib_stack.header["pixdim"] = [1.0, res[0], res[1], res[2], 0., 0., 0., 0.]
    save_file = os.path.join(save_folder, save_file_name)
    check_folder(save_file)
    nib.save(nib_stack, save_file)

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
    save_file = os.path.join(save_folder, save_file_name)
    check_folder(save_file)
    nib.save(nib_stack, save_file)

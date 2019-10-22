'''data augmentation'''
# import dependency library
import random
import math
from scipy.ndimage.morphology import binary_dilation
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt

#=============================================
#   augmentation for raw data
#=============================================


#=============================================
#   augmentation for label
#=============================================
#  get sliced distance transform which cross the center of nucleus.
def sliced_distance(label, center_stack=None, d_threshold=15):
    '''
    :param label:  Binary label of the target
    :param center_stack:
    :return:
    '''
    mask = np.ones_like(center_stack, dtype=bool)
    if center_stack is not None:
        nuc_x, nuc_y, nuc_z = np.nonzero(center_stack)
        mask[nuc_x, :, :] = False; mask[:, nuc_y, :] = False; mask[:, :, nuc_z] = False  #still too many slices annotation
    label[mask] = 0
    vertical_slice_edt = distance_transform_edt(label == 0)
    vertical_slice_edt[vertical_slice_edt > d_threshold] = d_threshold
    vertical_slice_edt = (d_threshold - vertical_slice_edt) / d_threshold
    vertical_slice_edt[mask] = 0

    return vertical_slice_edt.astype(np.float32)

#  cell centered distance transform
def cell_sliced_distance(seg_cell, seg_nuc, sampled=True, d_threshold=15):
    # sampled cell labels
    cell_labels = np.unique(seg_cell)[1:].tolist()
    sampled_num = 10 if len(cell_labels) > 10 else len(cell_labels)
    sampled_labels = random.sample(cell_labels, k=sampled_num)
    cell_mask = np.isin(seg_cell, sampled_labels)
    #  edt transformation
    vertical_slice_edt = distance_transform_edt(seg_nuc == 0)
    vertical_slice_edt[vertical_slice_edt > d_threshold] = d_threshold
    vertical_slice_edt = (d_threshold - vertical_slice_edt) / d_threshold
    # vertical_slice_edt = add_volume_boundary_mask(vertical_slice_edt, fill_value=0)
    # #  to simulate slice annotation, only keep slices through the nucleus # TODO: change from slice to entire cell mask
    # keep_mask = np.zeros_like(cell_mask, dtype=bool)
    # for label in sampled_labels:
    #     tem_mask = np.zeros_like(cell_mask, dtype=bool)
    #     single_cell_mask = (seg_cell==label)
    #     x, y, z = np.nonzero(np.logical_and(seg_nuc, single_cell_mask))
    #     tem_mask[x, :, :] = True; tem_mask[:, y, :] = True; tem_mask[:, :, z] = True  # still too many slices annotation
    #     # combine different cells
    #     keep_mask = np.logical_or(keep_mask, np.logical_and(single_cell_mask, tem_mask))
    #
    # vertical_slice_edt[~keep_mask] = 0  # Output -1 for less attetion in loss
    keep_mask = np.ones_like(vertical_slice_edt)
    return vertical_slice_edt.astype(np.float), (keep_mask).astype(np.float)   # output keep_mask to count on valid mask

#  sample partials binary cell boundary and cell mask for weak supervision
def sampled_cell_mask(seg_cell, seg_memb):
    cell_labels = np.unique(seg_cell)[1:].tolist()
    sampled_num = 200 if len(cell_labels) > 200 else len(cell_labels)
    sampled_labels = random.sample(cell_labels, k=sampled_num)
    cell_mask = np.isin(seg_cell, sampled_labels)
    strl = np.ones((3, 3, 3), dtype=bool)
    keep_mask = binary_dilation(cell_mask, structure=strl)
    cell_mask = np.logical_and(keep_mask, seg_memb)

    return cell_mask.astype(np.float), keep_mask.astype(np.float)

#  change regression data to discrete class data
def regression_to_class(res_data, out_class, uniform=True):
    bins = np.arange(out_class) / (out_class - 1)
    if not uniform:
        bins = np.sqrt(bins)

    return np.digitize(res_data, bins, right=True)

#  add prior information to boundary mask
def add_volume_boundary_mask(data, fill_value):
    W, H, D = data.shape
    data[[0, W-1], :, :] = fill_value
    data[:, [0, H-1], :] = fill_value
    data[:, :, [0, D-1]] = fill_value

    return data


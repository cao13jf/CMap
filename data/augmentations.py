'''data augmentation'''
# import dependency library
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
        mask[nuc_x, :, :] = False; mask[:, nuc_y, :] = False; mask[:, :, nuc_z] = False
    label[mask] = 0
    vertical_slice_edt = distance_transform_edt(mask==0)
    vertical_slice_edt[vertical_slice_edt > d_threshold] = d_threshold
    vertical_slice_edt = (d_threshold - vertical_slice_edt) / d_threshold
    vertical_slice_edt[mask] = 0

    return vertical_slice_edt.astype(np.float32)




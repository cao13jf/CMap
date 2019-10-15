'''library for reading or writing data'''

import os
import pickle
import numpy as np
import nibabel as nib


#==============================================
#  read files
#==============================================
#  load *.nii.gz volume
def nib_load(file_name):
    if not os.path.exists(file_name):
        raise IOError("Cannot file {}".format(file_name))
    return nib.load(file_name).get_data()


#==============================================
#  write files
#==============================================
#  write *.pkl files
def pkl_save(data, path):
    with open(path, "wb") as f:
        pickle.dump(data, f)


#==============================================
#  data process
#==============================================
def normalize3d(image, mask=None):
    assert len(image.shape) == 3, "Only support 3 D"
    assert image[0, 0, 0] == 0, "Background should be 0"
    if mask is not None:
        mask = (image > 0)
    mean = image[mask].mean()
    std = image[mask].std()
    image = image.astype(dtype=np.float32)
    image[mask] = (image[mask] - mean) / std
    return image

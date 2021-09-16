'''library for reading or writing data'''

import os
import pickle
import imageio
import numpy as np
import pandas as pd
import nibabel as nib


#==============================================
#  read files
#==============================================
#  load *.nii.gz volume
def nib_load(file_name):
    if not os.path.exists(file_name):
        raise IOError("Cannot find file {}".format(file_name))
    return nib.load(file_name).get_data()

def read_new_cd(cd_file):
    df_nuc = pd.read_csv(cd_file, lineterminator="\n")
    df_nuc[["cell", "time"]] = df_nuc["Cell & Time"].str.split(":", expand=True)
    df_nuc = df_nuc.rename(columns={"X (Pixel)":"x", "Y (Pixel)":"y", "Z (Pixel)\r":"z"})
    df_nuc = df_nuc.astype({"x":float, "y":float, "z":float, "time":int})

    return df_nuc

#==============================================
#  write files
#==============================================
#  write *.pkl files
def pkl_save(data, path):
    with open(path, "wb") as f:
        pickle.dump(data, f)

#  write *.nii.gz files
def nib_save(data, file_name):
    check_folder(file_name)
    return nib.save(nib.Nifti1Image(data, np.eye(4)), file_name)

#  write MembAndNuc
def img_save(image, file_name):
    check_folder(file_name)
    imageio.imwrite(file_name, image)

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


#===============================================
#  other utils
def check_folder(file_name):
    dir_name = os.path.dirname(file_name)
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
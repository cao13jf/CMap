import numpy as np
import pandas as pd

from utils.data_io import nib_load, nib_save

[x_imaging, y_imaging, z_imaging] = [512, 712, 92]

raw_nuc = pd.read_csv("191108plc1p1-t200-nuclei.txt", sep=",", header=None)

filtered_nuc = raw_nuc[raw_nuc.loc[:, 9] != ' ']
yxz0 = filtered_nuc.iloc[:, 5:8].copy().to_numpy()

raw_memb = nib_load("../dataset/test/191108plc1p1/RawMemb/191108plc1p1_002_rawMemb.nii.gz")
[x_raw, y_raw, z_raw] = raw_memb.shape
yxz = (yxz0 * np.array([y_raw, x_raw, z_raw]) / [y_imaging, x_imaging, z_imaging]).astype(np.uint16).T.tolist()

nuc_seg = np.zeros_like(raw_memb)
nuc_seg[yxz[1], yxz[0], yxz[2]] = 1
nuc_seg = np.flip(nuc_seg, axis=2)

nib_save(nuc_seg, "../191108plc1p1_200_nucSeg.nii.gz")
print("test")
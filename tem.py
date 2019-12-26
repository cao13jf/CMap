'''
For rename
'''
import os
import glob

target_folder = "dataset/train/170704plc1p1/SegNuc"
files = glob.glob(os.path.join(target_folder, "*.nii.gz"))
for file in files:
    base_name = os.path.basename(file)
    if 'T' in base_name:
        time = base_name.split('_')[1][1:]
        target_name = '_'.join([base_name.split('_')[0], time.zfill(3), base_name.split('_')[-1]])
        os.rename(file, os.path.join(target_folder, target_name))
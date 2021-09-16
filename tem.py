'''
For rename
'''
import os
import glob
import pickle
import pandas as pd
import numpy as np
# from tqdm import tqdm
# from utils.data_io import nib_load, nib_save
# from scipy.ndimage.morphology import binary_dilation

'''
target_folder = "dataset/train/170704plc1p1/SegNuc"
files = glob.glob(os.path.join(target_folder, "*.nii.gz"))
for file in files:.t
    base_name = os.path.basename(file)
    if 'T' in base_name:
        time = base_name.split('_')[1][1:]
        target_name = '_'.join([base_name.split('_')[0], time.zfill(3), base_name.split('_')[-1]])
        os.rename(file, os.path.join(target_folder, target_name))
'''


#=========================================
# stastical numbers
#=========================================

# embryo_name = "200326plc1p3"
#
# max_time = 220
# ace_file = os.path.join("./dataset/test", embryo_name, "CD" + embryo_name + ".csv")
# df_time = pd.read_csv(ace_file)
# cell_nums = []
# for t in range(1, max_time+1):
#     cell_nums.append(df_time[df_time["time"]==t].shape[0])
#
# save_file = os.path.join("./dataset/test", embryo_name, "cell_numbers.csv")
# if not os.path.isdir(os.path.dirname(save_file)):
#     os.makedirs(os.path.dirname(save_file))
# pd.DataFrame(cell_nums, index=range(1, max_time+1)).to_csv(save_file)
#
# print("test here")


# structure=np.ones((5, 5, 5))
'''combien membrane and nucleus'''
# tps = [60, 88, 96, 121, 126, 130, 139, 164, 172, 176, 183]
# for tp in tqdm(tps):
#     memb_file = "dataset/test/200113plc1p2/RawMemb/200113plc1p2_" + str(tp).zfill(3) + "_rawMemb.nii.gz"
#     nuc_file = "dataset/test/200113plc1p2/SegNuc/200113plc1p2_" + str(tp).zfill(3) + "_segNuc.nii.gz"
#     memb = nib_load(memb_file)
#
#     # for membrane
#     memb = (memb * 255.0 / float((memb.max() - memb.min()))).astype(np.uint8)
#     # for nucleus
#     nuc = nib_load(nuc_file)
#     nuc = binary_dilation(nuc, structure=np.ones((5, 5, 5)))
#     memb[nuc != 0] = 255
#     save_file = "./tem/MembAndNuc/200113plc1p2_" + str(tp).zfill(3) + "_rawMemb.nii.gz"
#     nib_save(memb, save_file)

# =================================================
# save name dictionary
# =================================================
# with open("./dataset/name_dictionary.txt", "rb") as f:
#     number_dict = pickle.load(f)
#
# pd_dict = pd.DataFrame({"Number": list(number_dict.keys()), "Cell Name": list(number_dict.values())})
# pd_dict.to_csv("./dataset/number_dictionary.csv", index=False, header=True)

# =======================
# Change cell mask
# =======================
from utils.data_io import nib_load, nib_save
from utils.post_lib import get_boundafry
from utils.shape_analysis import get_contact_area, get_contact_area_fast

aa = nib_load("/home/home/ProjectCode/LearningCell/MembProjectCode/output/191108plc1p1/SegCellTimeCombined/191108plc1p1_005_segCell.nii.gz")
bound = get_boundafry(aa == 1214, b_width=1)
pairs, contacts = get_contact_area(aa)
nib_save(bound, "./bound.nii.gz")

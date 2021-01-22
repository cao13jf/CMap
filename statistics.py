#===================================================
# collect evaluation information
#===================================================
import os
import pickle
from tqdm import tqdm
import numpy as np
import pandas as pd
import multiprocessing as mp

from utils.data_io import nib_load
from utils.ProcessLib import get_surface_area, get_contact_area


def analysis_shape_single_tp(para):
    embryo_name = para[0]
    t = para[1]
    name_dict = para[2]
    seg_file = os.path.join(seg_folder, "_".join([embryo_name, str(t).zfill(3), "segCell.nii.gz"]))
    seg = nib_load(seg_file)

    # volume
    cell_labels, cell_counts = np.unique(seg, return_counts=True)
    cell_labels = cell_labels[1:].tolist()
    cell_counts = cell_counts[1:].tolist()

    # surface
    cell_surface = [get_surface_area(seg == cell_label) for cell_label in cell_labels]

    # contact
    contact_pairs, contact_areas = get_contact_area(seg)
    contact_dict = {(contact_pair[0], contact_pair[1]):contact_area for contact_pair, contact_area in zip(contact_pairs, contact_areas)}

    cell_names = [name_dict[idx] for idx in cell_labels]
    cell_surfaces = dict(zip(cell_names, cell_surface))
    cell_volumes = dict(zip(cell_names, cell_counts))

    save_file = os.path.join("./statistics/tem", embryo_name, str(t).zfill(3) + "_" + embryo_name + ".txt")
    with open(save_file, "wb") as f:
        pickle.dump([cell_surfaces, cell_volumes, contact_dict], f)



if __name__ == "__main__":

    embryo_name = "200326plc1p4"
    max_time = 195
    save_folder = os.path.join("./statistics", embryo_name)
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)

    # load name dictionary
    with open("dataset/name_dictionary.txt", "rb") as f:
        name_dict = pickle.load(f)

    # =======================================================
    # stastical nucleus numbers
    # # =======================================================
    # ace_file = os.path.join("./dataset/test", embryo_name, "CD" + embryo_name + ".csv")
    # df_time = pd.read_csv(ace_file)
    # cell_nums = []
    # for t in range(1, max_time + 1):
    #     cell_nums.append([t, df_time[df_time["time"] == t].shape[0]])
    #
    # pd.DataFrame(cell_nums, columns=["Time", "Number of Cells"]).to_csv(os.path.join(save_folder, "cell_numbers.csv"), header=True, index=False)

    # ========================================================
    # volume distribution first, then threshold
    # ========================================================
    volume_threshold = 30
    seg_folder = os.path.join("output", embryo_name, "SegCell")
    tem_fodler = os.path.join("./statistics/tem", embryo_name)
    if not os.path.isdir(tem_fodler):
        os.makedirs(tem_fodler)

    cell_names = []
    cell_times = []
    cell_volumes = []
    pd_volume = pd.DataFrame(np.nan, index=range(1, max_time + 1), columns=name_dict.values())
    contact_dicts = {}
    paras = []
    for t in tqdm(range(1, max_time + 1)):
        paras.append([embryo_name, t, name_dict])
        # analysis_shape_single_tp(paras[0])
    mp_pool = mp.Pool(mp.cpu_count() - 2)
    for idx, _ in tqdm(enumerate(mp_pool.imap_unordered(analysis_shape_single_tp, paras)), total=len(paras), desc="Shape of {}".format(embryo_name)):
        pass

    # pd_volume.to_csv(os.path.join(save_folder, "cell_volume.csv"))
    # pd_surface = pd.DataFrame.from_dict(contact_dicts, orient="index")
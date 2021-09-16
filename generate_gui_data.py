import os
import glob
import pickle
import warnings
import shutil
import numpy as np
import nibabel as nib
from tqdm import tqdm
import pandas as pd
from csv import reader, writer

# import user defined library
from utils.data_io import nib_load, nib_save


# ***********************************************
# functions
# ***********************************************
def test_folder(folder_name):
    if "." in folder_name[1:]:
        folder_name = os.path.dirname(folder_name)
    if not os.path.isdir(folder_name):
        os.makedirs(folder_name)

def transpose_csv(source_file, target_file):
    with open(source_file) as f, open(target_file, 'w') as fw:
        writer(fw, delimiter=',').writerows(zip(*reader(f, delimiter=',')))

RENAME_FLAG = True
DELETE_FAILED_CELLS = False
CELLSPAN_FLAG = False
TP_CELLS_FLAGE = False
LOST_CELL = False
NEIGHBOR_FLAG = False
GET_DIVISIONS = False
CLEAR_SURFACE = False
COPY_FILE = False

embryo_names = ["200113plc1p2", "200113plc1p3", "200322plc1p2"]
# ==============================================================================
# generate data for GUI
# ==============================================================================
res_embryos = {0.25: [],
               0.18: ["191108plc1p1", "200109plc1p1", "200113plc1p2", "200113plc1p3", "200322plc1p2", "200323plc1p1",
                      "200326plc1p3", "200326plc1p4"],
               }

filtered_file = "./statistics/ToAnnotateListAll.csv"
failed_pd = pd.read_csv(filtered_file, header=0, index_col=None)
failed_pd = failed_pd[failed_pd["Death"] != 1]

pd_number = pd.read_csv('./dataset/number_dictionary.csv', names=["name", "label"])
number_dict = pd.Series(pd_number.label.values, index=pd_number.name).to_dict()
label2name_dict = dict((v, k) for k, v in number_dict.items())


# =================== Delete Failed Cells ======================================
if DELETE_FAILED_CELLS:
    seg_folder = "./output"
    annotate_to_save = seg_folder

    # ==== Multiple segs in the same file will be saved together
    old_embryo_name = ""
    old_time_point = ""
    labels = []
    FIRST_RUN = True
    for i_file, file_string in tqdm(failed_pd["File Info"].items(), desc="Separating segs"):
        embryo_name, time_point, label = file_string.split("_")
        label = int(label)
        if FIRST_RUN:
            old_embryo_name = embryo_name
            old_time_point = time_point
            FIRST_RUN = False

        if (old_time_point != time_point or old_embryo_name != embryo_name) and len(labels) != 0:

            # generate seg
            target_file = "_".join([old_embryo_name, old_time_point, "segCell.nii.gz"])
            file_name = os.path.join(seg_folder, old_embryo_name, "SegCellTimeCombinedLabelUnified", target_file)
            seg_cell = nib.load(file_name).get_fdata()
            for target_label in labels:
                seg_cell[seg_cell == target_label] = 0
            target_file_name = file_name  # os.path.join(annotate_to_save, old_embryo_name, target_file)
            nib_save(file_name=target_file_name, data=seg_cell)

            # update flags
            old_embryo_name = embryo_name
            old_time_point = time_point
            labels = [label]

        else:
            labels.append(label)

# =================================================
# write header (https://brainder.org/2012/09/23/the-nifti-file-format/)
# =================================================
if RENAME_FLAG:
    data_folder = "dataset/test"
    data_files = []
    for embryo_name in embryo_names:
        seg_folder = "./output/{}/SegCellTimeCombinedLabelUnified".format(embryo_name)
        data_files += glob.glob(os.path.join(data_folder, embryo_name, "*/*.nii.gz"), recursive=True)
        # data_files += glob.glob(os.path.join(seg_folder, "*.nii.gz"))
    data_files.sort()
    for data_file in tqdm(data_files, desc="Adding header"):
        img = nib.load(data_file).get_fdata().astype(np.uint8)
        img = nib.Nifti1Image(img, np.eye(4))
        img.header.set_xyzt_units(xyz=3, t=8)
        res_flag = False
        for res, embryos in res_embryos.items():
            if any([embryo in data_file for embryo in embryos]):
                res_flag = True
                img.header["pixdim"] = [1.0, res, res, res, 0., 0., 0., 0.]
                nib.save(img, data_file)
                break
        if not res_flag:
            warnings.warn("No resolution for {}!".format(data_file.split("/")[-1]))

# =================== save cell life span ======================================
if CELLSPAN_FLAG:
    for embryo_name in embryo_names:
        write_file = "./Tem/GUIData/{}/{}_lifescycle.csv".format(embryo_name, embryo_name)
        test_folder(write_file)

        # Collect lost life cycles of all cells  < --- updated here
        lost_cell_cycles = {}
        failed_pd_tp = failed_pd[failed_pd["Embryo Name"].str.contains(embryo_name)]
        tp_cell_strs = failed_pd_tp["File Info"].tolist()
        for tp_cell_str in tp_cell_strs:
            tp = int(tp_cell_str.split("_")[1])
            cell = int(tp_cell_str.split("_")[2])
            if int(cell) in lost_cell_cycles:
                lost_cell_cycles[cell] = lost_cell_cycles[cell] + [tp]
            else:
                lost_cell_cycles.update({cell: [tp]})

        with open("statistics/{}/{}_time_tree.txt".format(embryo_name, embryo_name), 'rb') as f:
            time_tree = pickle.load(f)
        all_cells = list(time_tree.nodes)
        i = 0
        for i, one_cell in enumerate(tqdm(all_cells, desc="Life span {}".format(embryo_name))):
            if i == 0:
                open(write_file, "w").close()
            times = time_tree.get_node(one_cell).data
            if times is None:
                continue
            if len(times.get_time()) == 0:
                continue
            if number_dict[one_cell] in lost_cell_cycles:
                time_nums = list(set(times.get_time()) - set(lost_cell_cycles[number_dict[one_cell]]))
            else:
                time_nums = times.get_time()

            times = [number_dict[one_cell]] + time_nums
            write_string = ",".join([str(x) for x in times]) + "\n"

            with open(write_file, "a") as f:
                f.write(write_string)


# # ======================== generate TP Cells =============================================
if TP_CELLS_FLAGE:
    for embryo_name in embryo_names:
        save_folder = "./Tem/GUIData/{}/TPCell".format(embryo_name, embryo_name)
        test_folder(save_folder)

        folder_name = os.path.join("./output", embryo_name, "SegCellTimeCombinedLabelUnified")
        file_list = glob.glob(os.path.join(folder_name, "*.nii.gz"))
        file_list.sort()
        for file_name in tqdm(file_list, desc="TP Cells {}".format(embryo_name)):
            seg = nib.load(file_name).get_fdata().astype(np.uint16)
            cell_labels = np.unique(seg).tolist()
            cell_labels.sort()
            cell_labels.remove(0)
            cell_string = ",".join(([str(cell_label) for cell_label in cell_labels]))

            base_name = os.path.basename(file_name)
            save_file = os.path.join(save_folder, "_".join(base_name.split("_")[:2]+["cells.txt"]))
            with open(save_file, "w") as f:
                f.write(cell_string+"\n")

# # get lost cells (added with filtered cells)
if LOST_CELL:
    for embryo_name in embryo_names:
        seg_folder = os.path.join("./output", embryo_name, "SegCellTimeCombined")
        # nucleus_folder = os.path.join("./Data/MembTest", embryo_name, "SegNuc")
        seg_files = sorted(glob.glob(os.path.join(seg_folder, "*.nii.gz")))
        # nucleus_files = glob.glob(os.path.join(nucleus_folder, "*.nii.gz"))
        seg_files.sort()
        # nucleus_files.sort()

        save_folder = "./Tem/GUIData/{}/LostCell".format(embryo_name, embryo_name)
        test_folder(save_folder)

        for i, seg_file in enumerate(tqdm(seg_files, desc="Lost cells {}".format(embryo_name)), start=1):
            base_name = os.path.basename(seg_file)
            target_str = "{}_{}".format(embryo_name, str(i).zfill(3))
            failed_pd_tp = failed_pd[failed_pd["File Info"].str.contains(target_str)]
            if len(failed_pd_tp.index) > 0:
                cell_string = [tem.split("_")[-1] for tem in failed_pd_tp["File Info"].tolist()]
                cell_string = ",".join(cell_string)
            else:
            # nucleus_file = nucleus_files[i]
            # seg = nib.load(seg_file).astype(np.uint16)
            # nucleus = nib.load(nucleus_file).astype(np.uint16)
            #
            # lost_cells = np.unique(nucleus[seg == 0]).tolist()
            # lost_cells.remove(0)
            # cell_string = ",".join(([str(cell_label) for cell_label in lost_cells]))

                cell_string = ''  # !!!!! Cell loss cannot be detected because of the seeded watershed
            save_file = os.path.join(save_folder, "_".join(base_name.split("_")[:2]+["lostCell.txt"]))
            with open(save_file, "w") as f:
                f.write(cell_string + "\n")


# # get neighboer
if NEIGHBOR_FLAG:
    for embryo_name in embryo_names:
        file_list = glob.glob(os.path.join('./statistics/TemCellGraph', embryo_name, embryo_name + '*.txt'))
        file_list = [file for file in file_list if "nucLoc" not in file]
        file_list.sort()
        for file_name in tqdm(file_list, desc="Getting neighbors {}".format(embryo_name)):
            with open(file_name, 'rb') as f:
                cell_graph = pickle.load(f)

            tp = os.path.basename(file_name).split("_")[1].split(".")[0][1:]
            base_name = "_".join([embryo_name, tp.zfill(3)])
            save_file = os.path.join("./Tem/GUIData", embryo_name, "GuiNeighbor", base_name+"_guiNeighbor.txt")
            test_folder(save_file)

            for i, cell_name in enumerate(cell_graph.nodes()):
                main_cell_str = "{}_{}_{}".format(embryo_name, tp.zfill(3), str(number_dict[cell_name]))
                if len(failed_pd[failed_pd["File Info"].str.contains(main_cell_str)]) > 0:
                    continue
                neighbor_cells = list(cell_graph.neighbors(cell_name))

                neighbor_labels0 = [str(number_dict[name]) for name in neighbor_cells]
                if len(neighbor_labels0) == 0:
                    continue
                neighbor_labels = []
                for neighbor_label in neighbor_labels0:
                    neighbor_cell_str = "{}_{}_{}".format(embryo_name, tp.zfill(3), neighbor_label)
                    if len(failed_pd[failed_pd["File Info"].str.contains(neighbor_cell_str)]) == 0:
                        neighbor_labels.append(neighbor_label)

                cell_label = str(number_dict[cell_name])
                label_str = ",".join(([cell_label] + neighbor_labels))  # first one master cell
                if i == 0:
                    with open(save_file, "w") as f:
                        f.write(label_str+"\n")
                else:
                    with open(save_file, "a") as f:
                        f.write(label_str+"\n")

# get cell divisions
if GET_DIVISIONS:
    for embryo_name in embryo_names:
        save_folder = os.path.join("./Tem/GUIData", embryo_name, "DivisionCell")
        test_folder(save_folder)

        loc_files = sorted(glob.glob("./ResultCell/NucleusLoc/{}/*.csv".format(embryo_name)))
        for tp, loc_file in enumerate(tqdm(loc_files, desc="Generating division of {}".format(embryo_name)), start=1):
            loc_pd = pd.read_csv(loc_file, header=0)
            division_pd = loc_pd[loc_pd["note"] == "mother"]
            cell_labels = division_pd["nucleus_label"].tolist()
            cell_trs = ",".join([str(i) for i in cell_labels])

            save_file = os.path.join(save_folder, "{}_{}_division.txt".format(embryo_name, str(tp).zfill(3)))
            with open(save_file, "w") as f:
                f.write(cell_trs + "\n")

# set surface and volume information of lost cells as zeros
if CLEAR_SURFACE:
    for embryo_name in tqdm(embryo_names, desc="Clear surface"):

        # change surface and volume
        failed_cell_strs = failed_pd[failed_pd["Embryo Name"] == embryo_name]["File Info"].tolist()
        tps = [int(failed_cell_str.split("_")[1])-1 for failed_cell_str in failed_cell_strs]
        cell_names = [label2name_dict[int(failed_cell_str.split("_")[-1])] for failed_cell_str in failed_cell_strs]
        # volume and surface
        surface_file = os.path.join("./statistics", embryo_name, embryo_name, embryo_name + "_surface.csv")
        volume_file = os.path.join(os.path.join("./statistics", embryo_name, embryo_name, embryo_name + "_volume.csv"))
        contact_file = os.path.join("./statistics", embryo_name, embryo_name, embryo_name + "_contact.csv")

        surface_pd = pd.read_csv(surface_file)
        volume_pd = pd.read_csv(volume_file)
        contact_pd = pd.read_csv(contact_file, header=[0, 1])
        for tp, cell_name in zip(tps, cell_names):
            surface_pd.at[tp, cell_name] = np.NaN
            volume_pd.at[tp, cell_name] = np.NaN
            try:
                contact_pd.loc[tp, (slice(None), cell_name)] = 0.0
                contact_pd.loc[tp, (cell_name, slice(None))] = 0.0
            except:
                pass

        # change the contact surface
        contact_pd = contact_pd.drop(cell_names, 1, level=0)
        contact_pd = contact_pd.drop(cell_names, 1, level=1)
        surface_pd.to_csv(os.path.join("./Tem/GUIData", embryo_name, embryo_name + "_surface.csv"))
        volume_pd.to_csv(os.path.join("./Tem/GUIData", embryo_name, embryo_name + "_volume.csv"))
        contact_pd.to_csv(os.path.join("./Tem/GUIData", embryo_name, embryo_name + "_Stat0.csv"), index=False)
        transpose_csv(os.path.join("./Tem/GUIData", embryo_name, embryo_name + "_Stat0.csv"),
                      os.path.join("./Tem/GUIData", embryo_name, embryo_name + "_Stat.csv"))



# ================== copy files ==============================
if COPY_FILE:
    # volume
    for embryo_name in embryo_names:
        shutil.copyfile(os.path.join("./statistics", embryo_name, embryo_name, embryo_name + "_surface.csv"),
               os.path.join("./Tem/GUIData", embryo_name, embryo_name + "_surface.csv"))
        # surface
        shutil.copyfile(os.path.join("./statistics", embryo_name, embryo_name, embryo_name + "_volume.csv"),
               os.path.join("./Tem/GUIData", embryo_name, embryo_name + "_volume.csv"))
        # contact (with transpose)
        transpose_csv(os.path.join("./statistics", embryo_name, embryo_name, embryo_name + "_contact.csv"),
               os.path.join("./Tem/GUIData", embryo_name, embryo_name + "_Stat.csv"))

    shutil.copyfile("./dataset/number_dictionary.csv", "./Tem/GUIData/name_dictionary.csv")
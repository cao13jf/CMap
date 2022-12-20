import os
import glob
import warnings
import numpy as np
import nibabel as nib
from tqdm import tqdm
import pandas as pd
from csv import reader, writer


# import user defined library
from utils.data_io import check_folder, move_file
from utils.cell_tree import construct_celltree, read_new_cd

# ***********************************************
# functions
# ***********************************************
def test_folder(folder_name):
    if "." in folder_name[1:]:
        folder_name = os.path.dirname(folder_name)
    if not os.path.isdir(folder_name):
        os.makedirs(folder_name)

def transpose_csv(source_file, target_file):
    with open(source_file) as f, open(target_file, 'w', newline='') as fw:
        writer(fw, delimiter=',').writerows(zip(*reader(f, delimiter=',')))

def generate_gui_data(args):

    RENAME_FLAG = True
    TP_CELLS_FLAGE = True
    CELLSPAN_FLAG = True
    NEIGHBOR_FLAG = True
    GET_DIVISIONS = True
    COPY_FILE_SEGMENTED = False
    COPY_STAT_FILE=True

    embryo_names = args.test_embryos

    # ==============================================================================
    # generate data for GUI
    # ==============================================================================
    z_resolutions = args.z_resolutions

    save_folder = args.gui_folder
    statistic_folder = "./statistics"
    seg_folder = "./output"
    raw_folder = args.test_data_dir
    name_file = "./dataset/number_dictionary.csv"

    pd_number = pd.read_csv(name_file, names=["name", "label"])
    number_dict = pd.Series(pd_number.label.values, index=pd_number.name).to_dict()
    name2label_dict = dict((k, v) for k, v in number_dict.items())

    all_lost_cells = []
    # ================== copy files ==============================
    if COPY_STAT_FILE:
        # volume
        for embryo_name in tqdm(embryo_names, desc="Moving stat files from SegCellTimeCombinedLabelUnifiedPost1"):
            check_folder(os.path.join(save_folder, embryo_name))
            move_file(os.path.join(statistic_folder, embryo_name, embryo_name + "_surface.csv"),
                            os.path.join(save_folder, embryo_name, embryo_name + "_surface.csv"))

            move_file(os.path.join(statistic_folder, embryo_name, embryo_name + "_volume.csv"),
                            os.path.join(save_folder, embryo_name, embryo_name + "_volume.csv"))
            # contact (with transpose)
            transpose_csv(os.path.join(statistic_folder, embryo_name, embryo_name + "_contact.csv"),
                          os.path.join(save_folder, embryo_name, embryo_name + "_Stat.csv"))
            if COPY_FILE_SEGMENTED:
                raw_folder0 = os.path.join(raw_folder, embryo_name, "RawMemb")
                raw_files = glob.glob(os.path.join(raw_folder0, "*.nii.gz"))
                seg_folder0 = os.path.join(seg_folder, embryo_name, "SegCellTimeCombinedLabelUnifiedPost1")
                seg_files = glob.glob(os.path.join(seg_folder0, "*.nii.gz"))
                save_file = os.path.join(raw_folder, embryo_name, "SegCell", os.path.basename(raw_files[0]))
                check_folder(save_file)
                save_file = os.path.join(raw_folder, embryo_name, "RawMemb", os.path.basename(raw_files[0]))
                check_folder(save_file)
                for raw_file, seg_file in zip(raw_files, seg_files):
                    save_file = os.path.join(save_folder, embryo_name, "RawMemb", os.path.basename(raw_file))
                    move_file(raw_file, save_file)
                    save_file = os.path.join(save_folder, embryo_name, "SegCell", os.path.basename(seg_file))
                    move_file(seg_file, save_file)

        pd_number = pd_number[["label", "name"]]
        pd_number = pd.concat([pd.DataFrame({"label": [np.NaN], "name": ["0"]}), pd_number], ignore_index=True)
        pd_number.to_csv(save_folder + "/name_dictionary.csv", index=False, header=False)

        CMapAddZeroToUnresonableBlank(args.gui_folder,args.gui_folder,embryo_names)

    # =================== save cell life span ======================================

    for embryo_name in embryo_names:
        print("Processing {} \n".format(embryo_name))

        volume_file = os.path.join(statistic_folder, embryo_name, embryo_name + "_volume.csv")
        contact_file = os.path.join(statistic_folder, embryo_name, embryo_name + "_contact.csv")
        ace_file = os.path.join(raw_folder, embryo_name, "CD{}.csv".format(embryo_name))

        volume_pd = pd.read_csv(volume_file, header=0, index_col=0)
        volume_pd.index = list(range(1, len(volume_pd.index) + 1, 1)) # todo:what is this?useless?
        contact_pd = pd.read_csv(contact_file, header=[0, 1], index_col=0)
        ace_pd = read_new_cd(ace_file)
        celltree, _ = construct_celltree(ace_file, len(volume_pd.index), name_file, read_old=False)

        # ----------------save cells at TPCell folder
        if TP_CELLS_FLAGE:
            bar = tqdm(total=len(volume_pd))
            bar.set_description("saving tp cells")
            for tp, row in volume_pd.iterrows():
                row = row.dropna()
                cell_names = list(row.index)
                cell_label = [name2label_dict[x] for x in cell_names] # transfer the cell name to cell label

                write_file = os.path.join(save_folder, embryo_name, "TPCell", "{}_{}_cells.txt".format(embryo_name, str(tp).zfill(3)))
                write_string = ",".join([str(x) for x in cell_label]) + "\n" # save the cell label for this tp
                check_folder(write_file)

                with open(write_file, "w") as f:
                    f.write(write_string)
                bar.update(1)


        # save ****_lifecycle.csv
        if CELLSPAN_FLAG:
            write_file = os.path.join(save_folder, embryo_name, "{}_lifescycle.csv".format(embryo_name))
            check_folder(write_file)

            open(write_file, "w").close()
            bar = tqdm(total=len(volume_pd.columns)) # go through volume csv
            bar.set_description("saving life cycle")
            for cell_col in volume_pd: # go through volume csv (cell name)
                valid_index = volume_pd[cell_col].notnull()
                tps = list(volume_pd[valid_index].index) # get the existing time point of this cell
                label_tps = [name2label_dict[cell_col]] + tps # combine the cell label and the existing time as a list

                write_string = ",".join([str(x) for x in label_tps]) + "\n"

                with open(write_file, "a") as f:
                    f.write(write_string)

            bar.update(1)

        # save neighbors ---GuiNeighbor
        if NEIGHBOR_FLAG:
            contact_pd = contact_pd.replace(0, np.nan)
            bar = tqdm(total=len(contact_pd))
            bar.set_description("saving neighbors")
            for tp, row in contact_pd.iterrows():
                row = row.dropna()
                neighbors = {}
                pairs = sorted(list(row.index))
                if len(pairs) == 0:
                    continue
                for cell1, cell2 in pairs:
                    cell1 = name2label_dict[cell1]
                    cell2 = name2label_dict[cell2]
                    if cell1 not in neighbors:
                        neighbors[cell1] = [cell2]
                    else:
                        neighbors[cell1] += [cell2]

                    if cell2 not in neighbors:
                        neighbors[cell2] = [cell1]
                    else:
                        neighbors[cell2] += [cell1]

                write_file = os.path.join(save_folder, embryo_name, "GuiNeighbor", "{}_{}_guiNeighbor.txt".format(embryo_name, str(tp).zfill(3)))
                check_folder(write_file)

                open(write_file, "w").close()
                with open(write_file, "a") as f:

                    for k, v in neighbors.items(): # cell label and its neighborhood
                        labels = [k] + list(set(v))
                        write_string = ",".join([str(x) for x in labels]) + "\n"
                        f.write(write_string)

                bar.update(1)


        # write division ------------ DivisionCell
        if GET_DIVISIONS:
            bar = tqdm(total=len(volume_pd))
            bar.set_description("saving neighbors")
            for tp, row in volume_pd.iterrows():
                row = row.dropna()
                cur_ace_pd = ace_pd[ace_pd["time"] == tp]
                nuc_cells = list(cur_ace_pd["cell"])  # cell in cd file this time point
                seg_cells = list(row.index)  # cell in volume.csv this time point
                dif_cells = list(set(nuc_cells) - set(seg_cells)) # only get the additional cells; lost cell?

                division_cells = []
                lost_cells = []

                # get average radius
                radii_mean = np.power(row, 1/3).mean()
                lost_radius = radii_mean * 1.3

                # if tp == 179:
                #     print("TEST")

                for dif_cell in dif_cells:
                    parent_cell = celltree.parent(dif_cell).tag
                    sister_cells = [x.tag for x in celltree.children(parent_cell)]
                    sister_cells.remove(dif_cell)
                    sister_cell = sister_cells[0]
                    # assert parent_cell in seg_cells
                    # division_cells.append(parent_cell)
                    if parent_cell in seg_cells:
                        division_cells.append(parent_cell)
                    else:
                        # seg_cell_file = os.path.join(save_folder, embryo_name, "SegCell", "{}_{}_segCell.nii.gz".format(embryo_name, str(tp).zfill(3)))
                        # seg = nib_load(seg_cell_file)
                        # sw, sh, sd = seg.shape
                        # nuc_loc = cur_ace_pd[cur_ace_pd["cell"] == dif_cell]
                        # locy, locx, locz = nuc_loc["x"].values[0] * sh / 712, nuc_loc["y"].values[0] * sw / 512, sd - nuc_loc["z"].values[0] * sd / max_slices[embryo_name]
                        # locx, locy, locz = int(locx), int(locy), int(locz)
                        # to_combine = seg[locx, locy, locz] == name2label_dict[sister_cell]
                        # # nuc_mask = np.zeros_like(seg, dtype=np.uint8)
                        # # nuc_mask = generate_sphere(seg, locx, locy, locz, lost_radius)
                        # # seg[nuc_mask != 0] = name2label_dict[dif_cell]
                        # # print(f"Add lost cells {seg_cell_file}")
                        # # nib_save(seg, seg_cell_file)
                        # # ================= add lost cell's point
                        # if to_combine:
                        #     lost_cell_name = "{}_{}_{}_{}_combine".format(embryo_name, str(tp).zfill(3), name2label_dict[dif_cell], dif_cell)
                        # else:
                        #     lost_cell_name = "{}_{}_{}_{}".format(embryo_name, str(tp).zfill(3), name2label_dict[dif_cell], dif_cell)

                        all_lost_cells.append("{}_{}_{}".format(embryo_name, dif_cell, str(tp).zfill(3)))
                        lost_cells.append(dif_cell)
                # each tp
                division_cells = list(set(division_cells))
                lost_cells = list(set(lost_cells))
                division_cells = [name2label_dict[x] for x in division_cells]
                lost_cells = [name2label_dict[x] for x in lost_cells]

                write_file = os.path.join(save_folder, embryo_name, "LostCell", "{}_{}_lostCell.txt".format(embryo_name, str(tp).zfill(3)))
                write_string = ",".join([str(x) for x in lost_cells]) + "\n"
                check_folder(write_file)

                with open(write_file, "w") as f:
                    f.write(write_string)

                write_file = os.path.join(save_folder, embryo_name, "DivisionCell", "{}_{}_division.txt".format(embryo_name, str(tp).zfill(3)))
                write_string = ",".join([str(x) for x in division_cells]) + "\n"
                check_folder(write_file)

                with open(write_file, "w") as f:
                    f.write(write_string)

                bar.update(1)

    # =================================================
    # write header (https://brainder.org/2012/09/23/the-nifti-file-format/)
    # =================================================
    if RENAME_FLAG:
        for res, embryo_name in zip(z_resolutions, embryo_names):
            data_files = glob.glob(os.path.join(save_folder, embryo_name, "*/*.nii.gz"), recursive=True) # go through all nii.gz files
            data_files.sort()
            for data_file in tqdm(data_files, desc="Adding header"):
                img = nib.load(data_file).get_fdata() # load
                img = nib.Nifti1Image(img, np.eye(4))
                img.header.set_xyzt_units(xyz=3, t=8)
                img.header["pixdim"] = [1.0, res, res, res, 0., 0., 0., 0.]
                nib.save(img, data_file) # then save to the same place??????????//


def CMapAddZeroToUnresonableBlank(gui_source_data_path,gui_target_data_path,embryo_names):

    # max_times = {"200117plc1pop1ip3":155}
    for embryo_name in embryo_names:
        contact_file=pd.read_csv(os.path.join(gui_source_data_path,embryo_name,embryo_name+'_Stat.csv'),header=0,index_col=[0,1])
        # for column_num,column_value in enumerate(contact_file.columns):
        #     print(column_num,column_value)
        # print(not contact_file.at[('Dap','Daaa'),str(144)]>=0)
        # print(not contact_file.loc[('Dap','Daaa')][143]>=0)
        # # Earap
        # # Earpp
        # print(not contact_file.at[('Earap','Earpp'),str(144)]>=0)
        #
        # input()
        for tp_index in contact_file.index:
            # print(embryo_name,'  contact pairs  ',tp_index)
            start_column=0
            stop_column=0
            first_flag=False
            # notNullIndex=contact_file.loc[tp_index].notna()
            for column_num,column_value in enumerate(contact_file.columns):
                # print(notNullIndex.loc[idx])
                if contact_file.at[tp_index,column_value]>=0 and not first_flag:
                    start_column=column_num
                    first_flag=True
                if contact_file.at[tp_index,column_value]>=0:
                    stop_column=column_num

            # print(start_column,stop_column)
            for col in range(start_column,stop_column+1):
                # if tp_index == ('Dap', 'Daaa'):
                #     # print(start_column, stop_column)
                #     print(col,not contact_file.loc[tp_index][col]>=0)
                # if contact_file.loc[tp_index][col]<0:
                #     print('gagagaga')
                if not contact_file.loc[tp_index][col] >=0:
                    # print(tp_index,col)
                    contact_file.loc[tp_index][col]=0
            # print(notNullIndex)
        contact_file.to_csv(os.path.join(gui_target_data_path,embryo_name,embryo_name+'_Stat.csv'))
import pickle
from csv import reader, writer
import os
import glob
from tqdm import tqdm
import shutil
import pandas as pd
import nibabel
import numpy as np
import io
from treelib import Tree

#  import user defined library
from utils.data_io import nib_load, normalize3d, pkl_save
from utils.post_lib import save_nii
from utils.data_structure import construct_celltree
from utils.shape_analysis import construct_stat_embryo


def CMap_stat_data_assemble():
    CMap_data_path = r'/home/home/ProjectCode/LearningCell/MembProjectCode'
    embryo_names = ['191108plc1p1', '200109plc1p1', '200113plc1p2', '200113plc1p3', '200322plc1p2', '200323plc1p1',
                    '200326plc1p3', '200326plc1p4', '200122plc1lag1ip1', '200122plc1lag1ip2', '200117plc1pop1ip2',
                    '200117plc1pop1ip3']

    max_times = [205, 205, 255, 195, 195, 185, 220, 195, 195, 195, 140, 155]
    # if not os.path.isdir('/home/jeff/ProjectCode/LearningCell/MembProjectCode/TemCellGraph'):
    #     os.makedirs('/home/jeff/ProjectCode/LearningCell/MembProjectCode/TemCellGraph')
    number_dictionary_path = os.path.join(
        '/home/jeff/ProjectCode/LearningCell/MembProjectCode/tem_files/tem_division_folder', 'name_dictionary.csv')
    pd_name_dict = pd.read_csv(number_dictionary_path, index_col=0, header=0)
    label_name = pd_name_dict.to_dict()['0']
    name_label = pd.Series(pd_name_dict.index, index=pd_name_dict['0']).to_dict()

    vol_coefficient = ((512 / 256) * 0.09) ** 3
    sur_coefficient = ((512 / 256) * 0.09) ** 2

    for idx, embryo_name in enumerate(embryo_names):
        cd_file_path = os.path.join(CMap_data_path, 'dataset/test', embryo_name, 'CD{}.csv'.format(embryo_name))
        cell_tree, max_time = construct_celltree(cd_file_path, max_times[idx])

        # ------------surface-------------and-----------volume-----------------------
        all_names = [cname for cname in cell_tree.expand_tree(mode=Tree.WIDTH)]
        # for idx, cell_name in enumerate(all_names):
        volume_embryo = pd.DataFrame(
            np.full(shape=(max_time, len(all_names)), fill_value=np.nan, dtype=np.float32),
            index=range(1, max_time + 1), columns=all_names)

        surface_embryo = pd.DataFrame(
            np.full(shape=(max_time, len(all_names)), fill_value=np.nan, dtype=np.float32),
            index=range(1, max_time + 1), columns=all_names)

        for tp in tqdm(range(1, max_time + 1),
                       desc='assembling volume and surface area of {} result'.format(embryo_name)):
            path_tmp = os.path.join(
                r'/home/home/ProjectCode/LearningCell/CellShapeAnalysis/DATA/cell_mesh_contact/stat', embryo_name)
            with open(os.path.join(path_tmp, '{}_{}_segCell_volume.txt'.format(embryo_name, str(tp).zfill(3))),
                      'rb') as handle:
                volume_dict = pickle.load(handle)
            with open(os.path.join(path_tmp, '{}_{}_segCell_surface.txt'.format(embryo_name, str(tp).zfill(3))),
                      'rb') as handle:
                surface_dict = pickle.load(handle)

            for cell_label_, vol_value in volume_dict.items():
                cell_name_ = label_name[cell_label_]
                volume_embryo.loc[tp, cell_name_] = vol_value * vol_coefficient

            for cell_label_, sur_value in surface_dict.items():
                cell_name_ = label_name[cell_label_]
                surface_embryo.loc[tp, cell_name_] = sur_value * sur_coefficient

        volume_embryo = volume_embryo.loc[:, ((volume_embryo != 0) & (~np.isnan(volume_embryo))).any(axis=0)]
        volume_embryo.to_csv(os.path.join(CMap_data_path, 'statistics', embryo_name, embryo_name + '_volume.csv'))
        surface_embryo = surface_embryo.loc[:, ((surface_embryo != 0) & (~np.isnan(surface_embryo))).any(axis=0)]
        surface_embryo.to_csv(os.path.join(CMap_data_path, 'statistics', embryo_name, embryo_name + '_surface.csv'))

        # ------------contact-----------initialize the contact csv  file----------------------
        # Get tuble lists with elements from the list
        # print(cell_tree)
        # print()
        name_combination = []
        first_level_names = []
        for i, name1 in enumerate(all_names):
            for name2 in all_names[i + 1:]:
                if not (cell_tree.is_ancestor(name1, name2) or cell_tree.is_ancestor(name2, name1)):
                    first_level_names.append(name1)
                    name_combination.append((name1, name2))

        multi_index = pd.MultiIndex.from_tuples(name_combination, names=['cell1', 'cell2'])
        # print(multi_index)
        stat_embryo = pd.DataFrame(
            np.full(shape=(max_time, len(name_combination)), fill_value=np.nan, dtype=np.float32),
            index=range(1, max_time + 1), columns=multi_index)
        # set zero element to express the exist of the specific nucleus
        for cell_name in all_names:
            if cell_name not in first_level_names:
                continue
            try:
                cell_time = cell_tree.get_node(cell_name).data.get_time()
                cell_time = [x for x in cell_time if x <= max_time]
                stat_embryo.loc[cell_time, (cell_name, slice(None))] = 0
            except:
                cell_name
        # print(stat_embryo)
        # edges_view = point_embryo.edges(data=True)
        for tp in tqdm(range(1, max_times[idx] + 1),
                       desc='assembling contact surface of {} result'.format(embryo_name)):
            path_tmp = os.path.join(
                r'/home/home/ProjectCode/LearningCell/CellShapeAnalysis/DATA/cell_mesh_contact/stat', embryo_name)
            with open(os.path.join(path_tmp, '{}_{}_segCell_contact.txt'.format(embryo_name, str(tp).zfill(3))),
                      'rb') as handle:
                contact_dict = pickle.load(handle)
            for contact_sur_idx, contact_sur_value in contact_dict.items():
                [cell1, cell2] = contact_sur_idx.split('_')
                cell1_name = label_name[int(cell1)]
                cell2_name = label_name[int(cell2)]
                if (cell1_name, cell2_name) in stat_embryo.columns:
                    stat_embryo.loc[tp, (cell1_name, cell2_name)] = contact_sur_value * sur_coefficient
                elif (cell2_name, cell1_name) in stat_embryo.columns:
                    stat_embryo.loc[tp, (cell2_name, cell1_name)] = contact_sur_value * sur_coefficient
                else:
                    pass
                    # print('columns missing (cell1_name, cell2_name)')
        stat_embryo = stat_embryo.loc[:, ((stat_embryo != 0) & (~np.isnan(stat_embryo))).any(axis=0)]

        print(stat_embryo)
        stat_embryo.to_csv(os.path.join(CMap_data_path, 'statistics', embryo_name, embryo_name + '_contact.csv'))
        # --------------------------------------------------------------------------------------------


def CShaper_data_assemble():
    CShaper_data_path = r'/home/home/ProjectCode/LearningCell/CellShapeAnalysis/DATA/Segmentation Results'
    embryo_names = ['Sample' + str(i).zfill(2) for i in range(4, 21)]
    max_times = [150, 170, 210, 165, 160, 160, 160, 170, 165, 150, 155, 170, 160, 160, 160, 160, 170]
    # if not os.path.isdir('/home/jeff/ProjectCode/LearningCell/MembProjectCode/TemCellGraph'):
    #     os.makedirs('/home/jeff/ProjectCode/LearningCell/MembProjectCode/TemCellGraph')
    number_dictionary_path = os.path.join(CShaper_data_path, 'name_dictionary.csv')
    pd_name_dict = pd.read_csv(number_dictionary_path, index_col=0, header=0)
    label_name = pd_name_dict.to_dict()['0']
    name_label = pd.Series(pd_name_dict.index, index=pd_name_dict['0']).to_dict()

    for idx, embryo_name in enumerate(embryo_names):
        cd_file_path = os.path.join(CShaper_data_path, 'RawData', embryo_name, 'CD{}.txt'.format(embryo_name))
        cell_tree, max_time = construct_celltree(cd_file_path, max_times[idx], read_cshaper_cd=True)

        # **************        # -----surface-------------and-----------volume-----------------------
        all_names = [cname for cname in cell_tree.expand_tree(mode=Tree.WIDTH)]
        # for idx, cell_name in enumerate(all_names):
        volume_embryo = pd.DataFrame(
            np.full(shape=(max_time, len(all_names)), fill_value=np.nan, dtype=np.float32),
            index=range(1, max_time + 1), columns=all_names)

        surface_embryo = pd.DataFrame(
            np.full(shape=(max_time, len(all_names)), fill_value=np.nan, dtype=np.float32),
            index=range(1, max_time + 1), columns=all_names)

        for tp in tqdm(range(1, max_time + 1),
                       desc='assembling volume and surface area of {} result'.format(embryo_name)):
            path_tmp = os.path.join(
                r'/home/home/ProjectCode/LearningCell/CellShapeAnalysis/DATA/cell_mesh_contact/stat', embryo_name)
            with open(os.path.join(path_tmp, '{}_{}_segCell_volume.txt'.format(embryo_name, str(tp).zfill(3))),
                      'rb') as handle:
                volume_dict = pickle.load(handle)
            with open(os.path.join(path_tmp, '{}_{}_segCell_surface.txt'.format(embryo_name, str(tp).zfill(3))),
                      'rb') as handle:
                surface_dict = pickle.load(handle)

            for cell_label_, vol_value in volume_dict.items():
                cell_name_ = label_name[cell_label_]
                volume_embryo.loc[
                    tp, cell_name_] = vol_value  # the volume and surface coefficient will be timesed in PROJECT GUI

            for cell_label_, sur_value in surface_dict.items():
                cell_name_ = label_name[cell_label_]
                surface_embryo.loc[tp, cell_name_] = sur_value

        volume_embryo = volume_embryo.loc[:, ((volume_embryo != 0) & (~np.isnan(volume_embryo))).any(axis=0)]
        volume_embryo.to_csv(os.path.join(CShaper_data_path, 'Stat', embryo_name + '_volume.csv'))
        surface_embryo = surface_embryo.loc[:, ((surface_embryo != 0) & (~np.isnan(surface_embryo))).any(axis=0)]
        surface_embryo.to_csv(os.path.join(CShaper_data_path, 'Stat', embryo_name + '_surface.csv'))

        # *******       # # ----contact--------------------------initialize the contact csv  file----------------------
        # Get tuble lists with elements from the list
        # print(cell_tree)
        # print()
        name_combination = []
        first_level_names = []
        for i, name1 in enumerate(all_names):
            for name2 in all_names[i + 1:]:
                if not (cell_tree.is_ancestor(name1, name2) or cell_tree.is_ancestor(name2, name1)):
                    first_level_names.append(name1)
                    name_combination.append((name1, name2))

        multi_index = pd.MultiIndex.from_tuples(name_combination, names=['cell1', 'cell2'])
        # print(multi_index)
        stat_embryo = pd.DataFrame(
            np.full(shape=(max_time, len(name_combination)), fill_value=np.nan, dtype=np.float32),
            index=range(1, max_time + 1), columns=multi_index)
        # set zero element to express the exist of the specific nucleus
        for cell_name in all_names:
            if cell_name not in first_level_names:
                continue
            try:
                cell_time = cell_tree.get_node(cell_name).data.get_time()
                cell_time = [x for x in cell_time if x <= max_time]
                stat_embryo.loc[cell_time, (cell_name, slice(None))] = 0
            except:
                cell_name
        # print(stat_embryo)
        # edges_view = point_embryo.edges(data=True)
        for tp in tqdm(range(1, max_times[idx] + 1),
                       desc='assembling contact surface of {} result'.format(embryo_name)):
            path_tmp = os.path.join(
                r'/home/home/ProjectCode/LearningCell/CellShapeAnalysis/DATA/cell_mesh_contact/stat', embryo_name)
            with open(os.path.join(path_tmp, '{}_{}_segCell_contact.txt'.format(embryo_name, str(tp).zfill(3))),
                      'rb') as handle:
                contact_dict = pickle.load(handle)
            for contact_sur_idx, contact_sur_value in contact_dict.items():
                [cell1, cell2] = contact_sur_idx.split('_')
                cell1_name = label_name[int(cell1)]
                cell2_name = label_name[int(cell2)]
                if (cell1_name, cell2_name) in stat_embryo.columns:
                    stat_embryo.loc[tp, (cell1_name, cell2_name)] = contact_sur_value
                elif (cell2_name, cell1_name) in stat_embryo.columns:
                    stat_embryo.loc[tp, (cell2_name, cell1_name)] = contact_sur_value
                else:
                    pass
                    # print('columns missing (cell1_name, cell2_name)')
        stat_embryo = stat_embryo.loc[:, ((stat_embryo != 0) & (~np.isnan(stat_embryo))).any(axis=0)]

        print(stat_embryo)
        stat_embryo.to_csv(os.path.join(CShaper_data_path, 'Stat', embryo_name + '_Stat.csv'))
        # --------------------------------------------------------------------------------------------


def CMap_nucloc_file_regenerator():
    CMap_data_path = r'/home/home/ProjectCode/LearningCell/MembProjectCode'
    embryo_names = ['191108plc1p1', '200109plc1p1', '200113plc1p2', '200113plc1p3', '200322plc1p2', '200323plc1p1',
                    '200326plc1p3', '200326plc1p4', '200122plc1lag1ip1', '200122plc1lag1ip2', '200117plc1pop1ip2',
                    '200117plc1pop1ip3']

    max_times = [205, 205, 255, 195, 195, 185, 220, 195, 195, 195, 140, 155]
    # if not os.path.isdir('/home/jeff/ProjectCode/LearningCell/MembProjectCode/TemCellGraph'):
    #     os.makedirs('/home/jeff/ProjectCode/LearningCell/MembProjectCode/TemCellGraph')
    number_dictionary_path = os.path.join(
        '/home/jeff/ProjectCode/LearningCell/MembProjectCode/tem_files/tem_division_folder', 'name_dictionary.csv')
    pd_name_dict = pd.read_csv(number_dictionary_path, index_col=0, header=0)
    label_name = pd_name_dict.to_dict()['0']
    name_label = pd.Series(pd_name_dict.index, index=pd_name_dict['0']).to_dict()

    volume_coefficient = ((512 / 256) * 0.09) ** 3
    surface_coefficient = ((512 / 256) * 0.09) ** 2

    for idx, embryo_name in enumerate(embryo_names):
        # ---------------------read cd file---------------------------------
        cd_file_path = os.path.join(CMap_data_path, 'dataset/test', embryo_name, 'CD{}.csv'.format(embryo_name))
        cd_file_all_dict = {}
        with io.open(cd_file_path, mode="r", encoding="utf-8") as f:
            # next(f)
            next(f)
            for line in f:
                info_list = line.split(',')
                [cellname_, tp_] = info_list[0].split(':')
                cd_file_all_dict[(cellname_, tp_)] = info_list[1:]
        print('constructing cell tree for ', embryo_name)
        cell_tree,_=construct_celltree(cd_file_path,max_time=max_times[idx])
        # ---------------------------------------------------------------
        for tp in tqdm(range(1, max_times[idx] + 1), desc='re-giving {} nucloc file'.format(embryo_name)):
            # unified_embryo_path = os.path.join(r'/home/home/ProjectCode/LearningCell/CellShapeAnalysis/DATA/',
            #                                    'Segmentation Results/UpdatedSegmentedCell', embryo_name,
            #                                    '{}_{}_segCell.nii.gz'.format(embryo_name, str(tp).zfill(3)))
            # unified_embryo = nib_load(unified_embryo_path)

            cd_dict_this = {}
            no_repeat_record_list=[]
            for (cell_name_tem, tp_tem), values_tem in cd_file_all_dict.items():
                # print(type(tp_tem),type(tp))
                if str(tp) == tp_tem:
                    cd_dict_this[cell_name_tem] = values_tem

            path_tmp = os.path.join(
                r'/home/home/ProjectCode/LearningCell/CellShapeAnalysis/DATA/cell_mesh_contact/stat',
                embryo_name)
            with open(os.path.join(path_tmp, '{}_{}_segCell_volume.txt'.format(embryo_name, str(tp).zfill(3))),
                      'rb') as handle:
                volume_dict = pickle.load(handle)
            with open(os.path.join(path_tmp, '{}_{}_segCell_surface.txt'.format(embryo_name, str(tp).zfill(3))),
                      'rb') as handle:
                surface_dict = pickle.load(handle)

            df_nucLoc = pd.DataFrame(
                columns=['nucleus_label','nucleus_name','x_256','y_356','z_214','volume','surface','note'])
            # for i in volume_dict.keys():
            #     print(i)
            #     print(type(i))
            for cell_name_, zxy_pos in cd_dict_this.items():
                # print(df_nucLoc.loc[df_nucLoc['nucleus_label']==cell_label_])

                cell_label_ = name_label[cell_name_]
                if cell_label_ in volume_dict.keys():
                    # print('normal CShaper makeup',embryo_name,tp,cell_name_)
                    # print(volume_dict)
                    # zxy_pos = cd_file_dict.get((cell_name_, str(tp)), None)

                    y = int(float(zxy_pos[0]) / 712 * 356)# labelled as X in cd file
                    x = int(float(zxy_pos[1]) / 512 * 256)# labelled as Y in cd file
                    z = 214 - int(float(zxy_pos[2]) / 92 * 214)

                    df_nucLoc.loc[len(df_nucLoc.index)] = [cell_label_, cell_name_, x, y, z,
                                                           volume_dict[cell_label_]*volume_coefficient, surface_dict[cell_label_]*surface_coefficient, None]
                    continue
                # -------------deal with dividing or lost cellsssssssssssssss----------------------------
                cell_mother_node = cell_tree.parent(cell_name_)
                cell_mother_name_=cell_mother_node.tag
                cell_mother_label_ = name_label.get(cell_mother_name_, None)
                if cell_mother_label_ in volume_dict.keys() and cell_mother_label_ not in no_repeat_record_list:
                    no_repeat_record_list.append(cell_mother_label_)

                    children_list=cell_tree.children(cell_mother_name_)
                    daughter_cell1=children_list[0].tag
                    daughter_cell2=children_list[1].tag
                    print('dividing ', embryo_name, tp, cell_mother_name_,daughter_cell1,daughter_cell2)

                    cell_label1=name_label[daughter_cell1]
                    cell_label2=name_label[daughter_cell2]

                    zxy_pos1 = cd_dict_this[daughter_cell1]
                    zxy_pos2 = cd_dict_this[daughter_cell2]

                    y1 = int(float(zxy_pos1[0]) / 712 * 356)# labelled as X in cd file
                    x1 = int(float(zxy_pos1[1]) / 512 * 256)# labelled as Y in cd file
                    z1 = 214 - int(float(zxy_pos1[2]) / 92 * 214)

                    y2 = int(float(zxy_pos2[0]) / 712 * 356)# labelled as X in cd file
                    x2 = int(float(zxy_pos2[1]) / 512 * 256)# labelled as Y in cd file
                    z2 = 214 - int(float(zxy_pos2[2]) / 92 * 214)

                    z = int((z1 + z2) / 2)
                    y = int((y1 + y2) / 2)  # labelled as X in cd file
                    x = int((x1 + x2) / 2)
                    df_nucLoc.loc[len(df_nucLoc.index)] = [cell_mother_label_, cell_mother_name_, x, y, z,
                                                           volume_dict[cell_mother_label_]*volume_coefficient,
                                                           surface_dict[cell_mother_label_]*surface_coefficient, 'mother']
                    df_nucLoc.loc[len(df_nucLoc.index)] = [cell_label1, daughter_cell1, x1,y1, z1,None,None,
                                                           'child of {}'.format(cell_mother_name_)]
                    df_nucLoc.loc[len(df_nucLoc.index)] = [cell_label2, daughter_cell2, x2,y2, z2,None,None,
                                                           'child of {}'.format(cell_mother_name_)]
                elif cell_mother_label_ not in no_repeat_record_list:
                    print('lost cell ', embryo_name, tp,cell_name_)
                    y = int(float(zxy_pos[0]) / 712 * 356)  # labelled as X in cd file
                    x = int(float(zxy_pos[1]) / 512 * 256)  # labelled as Y in cd file
                    z = 214 - int(float(zxy_pos[2]) / 92 * 214)

                    df_nucLoc.loc[len(df_nucLoc.index)] = [cell_label_, cell_name_, x, y, z,
                                                           None, None, 'lost']

            nucLoc_save_file_path = os.path.join(CMap_data_path, 'output/UpdatedNucleusLoc', embryo_name,
                                                 '{}_{}_nucLoc.csv'.format(embryo_name,
                                                                           str(tp).zfill(3)))
            df_nucLoc.to_csv(nucLoc_save_file_path, index=False)

def CShaper_data_makeup():
    CShaper_data_path = r'/home/home/ProjectCode/LearningCell/CellShapeAnalysis/DATA/Segmentation Results'
    embryo_names = ['Sample' + str(i).zfill(2) for i in range(4, 21)]
    max_times = [150, 170, 210, 165, 160, 160, 160, 170, 165, 150, 155, 170, 160, 160, 160, 160, 170]

    number_dictionary_path = os.path.join(CShaper_data_path, 'name_dictionary.csv')
    pd_name_dict = pd.read_csv(number_dictionary_path, index_col=0, header=0)
    label_name = pd_name_dict.to_dict()['0']
    name_label = pd.Series(pd_name_dict.index, index=pd_name_dict['0']).to_dict()

    wrong_division_list = []

    for idx, embryo_name in enumerate(embryo_names):
        embryo_this_path = os.path.join(CShaper_data_path, 'SegmentedCell', embryo_name + 'LabelUnified')
        cd_file_path = os.path.join(CShaper_data_path, 'RawData', embryo_name, 'CD{}.txt'.format(embryo_name))
        cd_file_dict = {}

        with io.open(cd_file_path, mode="r", encoding="utf-8") as f:
            # next(f)
            # next(f)
            for line in f:
                info_list = line.split()
                cd_file_dict[(info_list[0], info_list[1])] = info_list[2:]
        # print(cd_file_dict.keys())

        for tp in range(1, max_times[idx] + 1):
            seg_file = os.path.join(embryo_this_path, '{}_{}_segCell.nii.gz'.format(embryo_name, str(tp).zfill(3)))
            # print('working on segemented file ', '{}_{}_segCell.nii.gz'.format(embryo_name,str(tp).zfill(3)))
            flag_print = False
            embryo_this_tp = nibabel.load(seg_file).get_fdata().transpose([2, 1, 0])
            cell_list_this_tp = np.unique(embryo_this_tp)

            modifying_seg = np.copy(embryo_this_tp)
            for label_this_tp in cell_list_this_tp:
                if label_this_tp != 0:
                    name_this_tp = label_name[label_this_tp]

                    if (name_this_tp, str(tp)) in cd_file_dict.keys():
                        pass
                        # print(name_this_tp,cd_file_dict[(name_this_tp,str(tp))])
                        # zxy_pos = cd_file_dict[(name_this_tp, str(tp))]
                        # z = 114-int(float(zxy_pos[0]) / 68 * 114)
                        # x = int(float(zxy_pos[1]) / 712 * 256)
                        # y = int(float(zxy_pos[2]) / 512 * 184)
                        # print(embryo_this_tp.get_fdata().transpose([2,1,0]).shape)
                        # print(name_this_tp, '--mathching---',label_name[embryo_this_tp.get_fdata().transpose([2,1,0])[z, x, y]])
                    else:
                        # print(cd_file_dict.get((name_this_tp+'a',str(tp)),False),cd_file_dict.get((name_this_tp+'p',str(tp)),False),cd_file_dict.get((name_this_tp+'l',str(tp)),False),cd_file_dict.get((name_this_tp+'r',str(tp)),False))
                        # print(name_this_tp, 'is dividing')
                        if cd_file_dict.get((name_this_tp + 'a', str(tp)), False) and cd_file_dict.get(
                                (name_this_tp + 'p', str(tp)), False):
                            pass
                            # print(name_this_tp,' dividing as ', name_this_tp+'a',name_this_tp+'p',tp)
                        elif cd_file_dict.get((name_this_tp + 'l', str(tp)), False) and cd_file_dict.get(
                                (name_this_tp + 'r', str(tp)), False):
                            pass
                            # print(name_this_tp,' dividing as ', name_this_tp+'l',name_this_tp+'r',tp)
                        elif cd_file_dict.get((name_this_tp + 'd', str(tp)), False) and cd_file_dict.get(
                                (name_this_tp + 'v', str(tp)), False):
                            pass
                            # print(name_this_tp,' dividing as ', name_this_tp+'d',name_this_tp+'v',tp)
                        elif name_this_tp in ['EMS', 'P1', 'P2', 'P3', 'P4']:
                            pass
                        # elif cd_file_dict.get((name_this_tp+'aa',str(tp)),False) \
                        #         or cd_file_dict.get((name_this_tp+'ap',str(tp)),False) \
                        #         or cd_file_dict.get((name_this_tp+'pa',str(tp)),False) \
                        #         or cd_file_dict.get((name_this_tp+'pp',str(tp)),False):
                        #             print('{}_{}_segCell.nii.gz'.format(embryo_name, str(tp).zfill(3)), 'grandmother dividing cell ',(name_this_tp, str(tp)))
                        else:
                            print('{}_{}_segCell.nii.gz'.format(embryo_name, str(tp).zfill(3)), 'wrong dividing cell ',
                                  (name_this_tp, str(tp)))
                            for key, value in cd_file_dict.items():  # iter on both keys and values
                                # combine cells to their grandmother..
                                if key[0].startswith(name_this_tp) and key[1] == str(tp):
                                    print(name_this_tp, ' set to ', key[0], ' at time point ', tp)
                                    # key[0] str, name of the cell
                                    # key[1] str, time of the embryo
                                    modifying_seg[embryo_this_tp == name_label[name_this_tp]] = name_label[key[0]]
                                    wrong_division_list.append([embryo_name, tp, name_this_tp, key[0]])
                                    break

            # if flag_print:
            #     print(np.unique(embryo_this_tp),np.unique(modifying_seg))
            save_nii(modifying_seg, os.path.join(CShaper_data_path, 'UpdatedSegmentedCell', embryo_name,
                                                 '{}_{}_segCell.nii.gz'.format(embryo_name, str(tp).zfill(3))))
    print(wrong_division_list)
    with open(os.path.join('tem_files', 'CShaper_wrong_division_cells.pikcle'), 'wb') as fp:
        pickle.dump(wrong_division_list, fp)


def CShaper_nucloc_file_regenerate():
    CShaper_data_path = r'/home/home/ProjectCode/LearningCell/CellShapeAnalysis/DATA/Segmentation Results'
    embryo_names = ['Sample' + str(i).zfill(2) for i in range(4, 21)]
    max_times = [150, 170, 210, 165, 160, 160, 160, 170, 165, 150, 155, 170, 160, 160, 160, 160, 170]

    number_dictionary_path = os.path.join(CShaper_data_path, 'name_dictionary.csv')
    pd_name_dict = pd.read_csv(number_dictionary_path, index_col=0, header=0)
    label_name = pd_name_dict.to_dict()['0']
    name_label = pd.Series(pd_name_dict.index, index=pd_name_dict['0']).to_dict()

    volume_coefficient = 0.25 ** 3
    surface_coefficient = 0.25 ** 2
    for idx, embryo_name in enumerate(embryo_names):
        # ---------------------read cd file---------------------------------
        cd_file_path = os.path.join(CShaper_data_path, 'RawData', embryo_name,
                                    'CD{}.txt'.format(embryo_name))
        cd_file_all_dict = {}
        with io.open(cd_file_path, mode="r", encoding="utf-8") as f:
            # next(f)
            # next(f)
            for line in f:
                info_list = line.split()
                cd_file_all_dict[(info_list[0], info_list[1])] = info_list[2:]
        print('constructing cell tree for ', embryo_name)
        cell_tree,_=construct_celltree(cd_file_path,max_time=max_times[idx],read_cshaper_cd=True)
        # ---------------------------------------------------------------
        for tp in tqdm(range(1, max_times[idx] + 1), desc='re-giving {} nucloc file'.format(embryo_name)):
            # unified_embryo_path = os.path.join(r'/home/home/ProjectCode/LearningCell/CellShapeAnalysis/DATA/',
            #                                    'Segmentation Results/UpdatedSegmentedCell', embryo_name,
            #                                    '{}_{}_segCell.nii.gz'.format(embryo_name, str(tp).zfill(3)))
            # unified_embryo = nib_load(unified_embryo_path)

            cd_dict_this = {}
            no_repeat_record_list=[]
            for (cell_name_tem, tp_tem), values_tem in cd_file_all_dict.items():
                # print(type(tp_tem),type(tp))
                if str(tp) == tp_tem:
                    cd_dict_this[cell_name_tem] = values_tem

            path_tmp = os.path.join(
                r'/home/home/ProjectCode/LearningCell/CellShapeAnalysis/DATA/cell_mesh_contact/stat',
                embryo_name)
            with open(os.path.join(path_tmp, '{}_{}_segCell_volume.txt'.format(embryo_name, str(tp).zfill(3))),
                      'rb') as handle:
                volume_dict = pickle.load(handle)
            with open(os.path.join(path_tmp, '{}_{}_segCell_surface.txt'.format(embryo_name, str(tp).zfill(3))),
                      'rb') as handle:
                surface_dict = pickle.load(handle)

            df_nucLoc = pd.DataFrame(
                columns=['nucleus_label', 'nucleus_name', 'x_184', 'y_256', 'z_114', 'volume', 'surface', 'note'])
            # for i in volume_dict.keys():
            #     print(i)
            #     print(type(i))
            for cell_name_, zxy_pos in cd_dict_this.items():
                # print(df_nucLoc.loc[df_nucLoc['nucleus_label']==cell_label_])

                cell_label_ = name_label[cell_name_]
                if cell_label_ in volume_dict.keys():
                    # print('normal CShaper makeup',embryo_name,tp,cell_name_)
                    # print(volume_dict)
                    # zxy_pos = cd_file_dict.get((cell_name_, str(tp)), None)
                    z = 114 - int(float(zxy_pos[0]) / 68 * 114)
                    y = int(float(zxy_pos[1]) / 712 * 256)  # labelled as X in cd file
                    x = int(float(zxy_pos[2]) / 512 * 184)  # labelled as Y in cd file

                    df_nucLoc.loc[len(df_nucLoc.index)] = [cell_label_, cell_name_, x, y, z,
                                                           volume_dict[cell_label_]*volume_coefficient, surface_dict[cell_label_]*surface_coefficient, None]
                    continue
                # -------------deal with dividing or lost cellsssssssssssssss----------------------------
                cell_mother_node = cell_tree.parent(cell_name_)
                cell_mother_name_=cell_mother_node.tag
                cell_mother_label_ = name_label.get(cell_mother_name_, None)
                if cell_mother_label_ in volume_dict.keys() and cell_mother_label_ not in no_repeat_record_list:
                    no_repeat_record_list.append(cell_mother_label_)

                    children_list=cell_tree.children(cell_mother_name_)
                    daughter_cell1=children_list[0].tag
                    daughter_cell2=children_list[1].tag
                    print('dividing ', embryo_name, tp, cell_mother_name_,daughter_cell1,daughter_cell2)

                    cell_label1=name_label[daughter_cell1]
                    cell_label2=name_label[daughter_cell2]

                    zxy_pos1 = cd_dict_this[daughter_cell1]
                    zxy_pos2 = cd_dict_this[daughter_cell2]
                    z1 = 114 - int(float(zxy_pos1[0]) / 68 * 114)
                    y1 = int(float(zxy_pos1[1]) / 712 * 256)  # labelled as X in cd file
                    x1 = int(float(zxy_pos1[2]) / 512 * 184)  # labelled as Y in cd file

                    z2 = 114 - int(float(zxy_pos2[0]) / 68 * 114)
                    y2 = int(float(zxy_pos2[1]) / 712 * 256)  # labelled as X in cd file
                    x2 = int(float(zxy_pos2[2]) / 512 * 184)  # labelled as Y in cd file

                    z = int((z1 + z2) / 2)
                    y = int((y1 + y2) / 2)  # labelled as X in cd file
                    x = int((x1 + x2) / 2)
                    df_nucLoc.loc[len(df_nucLoc.index)] = [cell_mother_label_, cell_mother_name_, x, y, z,
                                                           volume_dict[cell_mother_label_]*volume_coefficient,
                                                           surface_dict[cell_mother_label_]*surface_coefficient, 'mother']
                    df_nucLoc.loc[len(df_nucLoc.index)] = [cell_label1, daughter_cell1, x1,y1, z1,None,None,
                                                           'child of {}'.format(cell_mother_name_)]
                    df_nucLoc.loc[len(df_nucLoc.index)] = [cell_label2, daughter_cell2, x2,y2, z2,None,None,
                                                           'child of {}'.format(cell_mother_name_)]
                elif cell_mother_label_ not in no_repeat_record_list:
                    print('lost cell ', embryo_name, tp,cell_name_)
                    z = 114 - int(float(zxy_pos[0]) / 68 * 114)
                    y = int(float(zxy_pos[1]) / 712 * 256)  # labelled as X in cd file
                    x = int(float(zxy_pos[2]) / 512 * 184)  # labelled as Y in cd file

                    df_nucLoc.loc[len(df_nucLoc.index)] = [cell_label_, cell_name_, x, y, z,
                                                           None, None, 'lost']

                # print(df_nucLoc.loc[df_nucLoc['nucleus_label']==cell_label_])
            #
            #
            # for cell_label_, sur_value in surface_dict.items():
            #     df_nucLoc.loc[df_nucLoc['nucleus_label']==cell_label_,'surface'] = sur_value*surface_coefficient
            nucLoc_save_file_path = os.path.join(CShaper_data_path, 'UpdatedNucLocFile', embryo_name,
                                                 '{}_{}_nucLoc.csv'.format(embryo_name,
                                                                           str(tp).zfill(3)))
            df_nucLoc.to_csv(nucLoc_save_file_path, index=False)


def move_and_build_CShaper_data_structure():
    """
    for shape analysis and label re-assignment only! will move them to output
    :return:
    """
    CShaper_data_path = r'/home/home/ProjectCode/LearningCell/CellShapeAnalysis/DATA/Segmentation Results/SegmentedCell'

    embryo_names = ['Sample' + str(i).zfill(2) for i in range(4, 21)]
    max_times = [150, 170, 210, 165, 160, 160, 160, 170, 165, 150, 155, 170, 160, 160, 160, 160, 170]

    for idx, embryo_name in enumerate(embryo_names):
        embryo_this_path_dst = os.path.join(r'/home/home/ProjectCode/LearningCell/MembProjectCode/output', embryo_name,
                                            'SegCellTimeCombined')
        os.makedirs(embryo_this_path_dst)
        for tp in range(1, max_times[idx] + 1):
            file_src = os.path.join(CShaper_data_path, embryo_name + 'LabelUnified',
                                    '{}_{}_segCell.nii.gz'.format(embryo_name, str(tp).zfill(3)))
            file_dst = os.path.join(embryo_this_path_dst, '{}_{}_segCell.nii.gz'.format(embryo_name, str(tp).zfill(3)))
            shutil.copy(file_src, file_dst)


def transpose_csv(source_file, target_file):
    with open(source_file) as f, open(target_file, 'w', newline='') as fw:
        writer(fw, delimiter=',').writerows(zip(*reader(f, delimiter=',')))


if __name__ == "__main__":
    # embryo_names = ["191108plc1p1", "200109plc1p1", "200113plc1p2", "200113plc1p3", "200322plc1p2", "200323plc1p1", "200326plc1p3", "200326plc1p4"]
    # max_times = [205, 205, 255, 195, 195, 185, 220, 195]
    # embryo_names = ["200117plc1pop1ip2", "200117plc1pop1ip3"]
    # max_times = [140, 155]

    # embryo_names = ["220501plc1p4"]
    # max_times = [240]
    # # doit(train_folder, embryo_names)
    # # dataset folder
    # # train_folder = dict(root="dataset/train", has_label=True)
    # test_folder = dict(root="dataset/test", has_label=False)
    # doit(test_folder, embryo_names, max_times=max_times)

    # move_and_build_CShaper_data_structure()
    # CShaper_nucloc_file_regenerate()
    # CShaper_data_assemble()
    # CShaper_data_makeup()
    # CMap_stat_data_assemble()
    CMap_nucloc_file_regenerator()

# import dependency library
import sys
import shutil
import warnings
from copy import deepcopy

import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy import ndimage
import multiprocessing as mp
from skimage import morphology
from skimage.measure import marching_cubes_lewiner, mesh_surface_area

# import user defined library
from utils.alpha_shape import generate_alpha_shape
from utils.draw_lib import *
from utils.data_structure import *
from utils.post_lib import check_folder_exist
from utils.post_lib import save_nii, get_boundafry, get_contact_pairs
from utils.parse_config import parse_config
from utils.data_io import read_new_cd

warnings.filterwarnings("ignore")

stat_embryo = None  # Global embryo shape information
max_time = None
cell_tree = None


def init(l):  # used for parallel computing
    global file_lock
    file_lock = l


# global file_block
# file_lock = None


def shape_analysis_func(args):
    max_times = args.max_times
    embryo_names = args.test_embryos
    raw_size = args.raw_size

    for i_embryo, embryo_name in enumerate(embryo_names):
        max_time = max_times[i_embryo]
        # Construct folder
        para_config = {}
        para_config["xy_resolution"] = args.xy_resolution
        para_config["max_time"] = max_time
        para_config["embryo_name"] = embryo_name
        # para_config["data_folder"] = os.path.join("dataset/test", embryo_name)
        para_config["save_nucleus_folder"] = "output/NucleusLoc"
        para_config["seg_folder"] = os.path.join("output", embryo_name, "SegCellTimeCombined")
        para_config["stat_folder"] = os.path.join("statistics", embryo_name)
        para_config["delete_tem_file"] = False
        # para_config["num_slice"] = raw_size[0]
        para_config["acetree_file"] = os.path.join("./dataset/test", embryo_name, "".join(["CD", embryo_name, ".csv"]))
        para_config["project_folder"] = "./statistics"
        para_config["number_dictionary"] = "dataset/number_dictionary.csv"

        para_config['mesh_path'] = r'/home/home/ProjectCode/LearningCell/CellShapeAnalysis/DATA/cell_mesh_contact/'

        if not os.path.isdir(para_config['stat_folder']):
            os.makedirs(para_config['stat_folder'])

        # Get the size of the figure
        # example_embryo_folder = os.path.join(para_config["raw_folder"], para_config["embryo_name"], "tif")
        # example_img_file = glob.glob(os.path.join(example_embryo_folder, "*.tif"))
        # raw_size = [para_config["num_slice"]] + list(np.asarray(Image.open(example_img_file[0])).shape)
        para_config["image_size"] = raw_size

        if not os.path.isdir(os.path.join(para_config['save_nucleus_folder'], para_config['embryo_name'])):
            os.makedirs(os.path.join(para_config['save_nucleus_folder'], para_config['embryo_name']))
        else:
            shutil.rmtree(os.path.join(para_config['save_nucleus_folder'], para_config['embryo_name']))
            os.makedirs(os.path.join(para_config['save_nucleus_folder'], para_config['embryo_name']))
        run_shape_analysis(para_config)

def run_shape_analysis(config):
    '''
    Extract the cell tree structure from the aceTree file
    :param acetree_file:  file name of the embryo acetree file
    :param max_time:  the maximum time point in the tree.
    :return :
    '''
    global max_time
    global cell_tree
    ## construct lineage tree whose nodes contain the time points that cell exist (based on nucleus).
    acetree_file = config['acetree_file']
    cell_tree, max_time = construct_celltree(acetree_file, config["max_time"])
    save_file_name = os.path.join(config['stat_folder'], config['embryo_name'] + '_time_tree.txt')

    with open(save_file_name, 'wb') as f:
        pickle.dump(cell_tree, f)
    ## Parallel computing for the cell relation graph
    if not os.path.isdir(os.path.join(config["project_folder"], 'TemCellGraph')):
        os.makedirs(os.path.join(config["project_folder"], 'TemCellGraph'))

    pd_number = pd.read_csv(config["number_dictionary"], names=["name", "label"])
    number_dict = pd.Series(pd_number.label.values, index=pd_number.name).to_dict()

    # ========================================================
    #       sementing TPs in a parallel way
    # ========================================================

    file_lock = mp.Lock()  # |-----> for change treelib files
    # print(file_lock, mp.cpu_count(), init)
    mpPool = mp.Pool(mp.cpu_count() - 1, initializer=init, initargs=(file_lock,))
    # mpPool = mp.Pool(10, initializer=init, initargs=(file_lock,))

    configs = []
    config["cell_tree"] = cell_tree

    # max_time=179
    for itime in tqdm(range(1, max_time + 1), desc="Compose configs"):
        config['time_point'] = itime
        configs.append(config.copy())

        # # -------------------- single test ---------------------------------
        # config['time_point'] = 50
        # cell_graph_network(config)
        # # -------------------- single test ---------------------------------

    embryo_name = config["embryo_name"]
    for idx, _ in enumerate(tqdm(mpPool.imap_unordered(cell_graph_network, configs), total=len(configs),
                                 desc="Naming {} segmentations (contact graph)".format(embryo_name))):
        #
        pass

    # ========================================================
    #       Combine previous TPs
    # ========================================================
    # ## In order to make use of parallel computing, the global vairable stat_embryo cannot be shared between different processor,
    # #  so we need to store all one-embryo reults as temporary files, which will be assembled finally. After that, these temporary
    # #  Data can be deleted.
    construct_stat_embryo(cell_tree,
                          max_time)  # initilize the shape matrix which is use to store the shape series information
    for itime in tqdm(range(1, max_time + 1), desc='assembling {} result'.format(embryo_name)):
        file_name = os.path.join(config["project_folder"], 'TemCellGraph', config['embryo_name'],
                                 config['embryo_name'] + '_T' + str(itime) + '.txt')
        with open(file_name, 'rb') as f:
            cell_graph = pickle.load(f)
            stat_embryo = assemble_result(cell_graph, itime, number_dict) # use saved cell contact graph to assemble result

    # # =======================================================
    # # Combine all surfaces and volumes in one single file
    # # =======================================================
    # # combien all volume and surface informace
    # volume_lists = []
    # surface_lists = []
    # for t in tqdm(range(1, max_time + 1), desc="Generate surface and volume {}".format(embryo_name.split('/')[-1])):
    #     nucleus_loc_file = os.path.join(config["save_nucleus_folder"], embryo_name, os.path.basename(embryo_name)+"_"+str(t).zfill(3)+"_nucLoc"+".csv")
    #     pd_loc = pd.read_csv(nucleus_loc_file)
    #     cell_volume_surface = pd_loc[["nucleus_name", "volume", "surface"]]
    #     cell_volume_surface = cell_volume_surface.set_index("nucleus_name")
    #     volume_lists.append(cell_volume_surface["volume"].to_frame().T.dropna(axis=1))
    #     surface_lists.append(cell_volume_surface["surface"].to_frame().T.dropna(axis=1))
    # volume_stat = pd.concat(volume_lists, keys=range(1, max_time+1), ignore_index=True, axis=0, sort=False, join="outer")
    # surface_stat = pd.concat(surface_lists, keys=range(1, max_time+1), ignore_index=True, axis=0, sort=False, join="outer")
    # volume_stat.to_csv(os.path.join(config["stat_folder"], embryo_name.split('/')[-1] + "_volume"+'.csv'))
    # surface_stat.to_csv(os.path.join(config["stat_folder"], embryo_name.split('/')[-1] + "_surface"+'.csv'))
    #
    #
    # if config['delete_tem_file']:  # If need to delete temporary files.
    #     shutil.rmtree(os.path.join(config["project_folder"], 'TemCellGraph'))

    stat_embryo = stat_embryo.loc[:, ((stat_embryo != 0) & (~np.isnan(stat_embryo))).any(axis=0)]
    save_file_name_csv = os.path.join(config['stat_folder'], config['embryo_name'] + '_contact.csv')
    stat_embryo.to_csv(save_file_name_csv)

    #### compose volume and surface information
    compose_surface_and_volume(embryo_name)


def cell_graph_network(config):
    '''
    Used to construct the contact relationship at one specific time point. The vertex represents the cell, and there is
    a edge whenever two SegCell contact with each other.
    :param config: parameter configs
    :return :
    '''
    time_point = config['time_point']
    # seg file is the "SegCellTimeCombined" FOld
    seg_file = os.path.join(config['seg_folder'], #para_config["seg_folder"] = os.path.join("output", embryo_name, "SegCellTimeCombined")
                            config['embryo_name'] + "_" + str(time_point).zfill(3) + '_segCell.nii.gz')
    nucleus_loc_file = os.path.join(config["project_folder"], 'TemCellGraph', config['embryo_name'], #para_config["project_folder"] = "./statistics"
                                    config['embryo_name'] + "_" + str(time_point).zfill(
                                        3) + '_nucLoc' + '.txt')  # read nucleus location Data

    #  Load the dictionary of cell and it's coresponding number in the dictionary
    pd_number = pd.read_csv(config["number_dictionary"], names=["name", "label"])
    number_dict = pd.Series(pd_number.label.values, index=pd_number.name).to_dict()
    name_dict = {value: key for key, value in number_dict.items()}

    ## unify the labels in the segmentation and that in the aceTree information
    division_seg, nuc_position, config["res"] = unify_label_seg_and_nuclues(file_lock, time_point, seg_file, config)
    # TODO: dangerous action 2 - I change 'LabelUnified' to 'LabelUnifiedPost1'
    division_seg_save_file = os.path.join(os.path.dirname(seg_file) + 'LabelUnified',
                                          config['embryo_name'] + "_" + str(time_point).zfill(3) + '_segCell.nii.gz')
    save_nii(division_seg, division_seg_save_file)

    ##  cinstruct graph
    #  add vertex
    with open(nucleus_loc_file, 'rb') as f:
        nucleus_loc = pickle.load(f)
    all_labels = list(np.unique(division_seg))
    all_labels.remove(0)
    point_graph = nx.Graph()
    point_graph.clear()
    for label in all_labels:
        cell_name = name_dict[label]
        point_graph.add_node(cell_name,
                             pos=nucleus_loc[nucleus_loc.nucleus_name == cell_name].iloc[:, 2:5].values[0].tolist())
    #  add connections between SegCell (edge and edge weight)
    # config["res"] = ace_shape[-1] / seg.shape[-1] * config["xy_resolution"]
    relation_graph = add_relation(point_graph, division_seg, name_dict, res=config["res"],
                                  config_this={'mesh_path':config['mesh_path'],'embryo_name':config['embryo_name'],'time_point':str(config['time_point']).zfill(3)})

    # nx.draw(relation_graph, pos=nuc_position, with_labels=True, node_size=100, font_color='b', edge_cmap=plt.cm.Blues)
    file_name = os.path.join(config["project_folder"], 'TemCellGraph', config['embryo_name'],
                             config['embryo_name'] + '_T' + str(config['time_point']) + '.txt')
    with open(file_name, 'wb') as f:
        pickle.dump(relation_graph, f)


def unify_label_seg_and_nuclues(file_lock, time_point, seg_file, config):
    '''
    Use acetree nucleus information to unify the segmentation labels in the membrane segmentations.
    :param file_lock: file locker to control parallel computing
    :param time_point: time point of the volume
    :param seg_file: cell segmentation file
    :param config: parameter configs
    :return unify_seg: cell segmentation with labels unified
    :return nuc_positions: nucleus positions with labels
    '''

    # todo: Danger usage 1 - here. But time is limited, use this first
    # list_wrong_division_label=[]
    # try:
    #     with open(os.path.join('tem_files', 'wrong_division_cells.pikcle'), "rb") as fp:  # Unpickling
    #         list_wrong_division_label = pickle.load(fp)
    # except:
    #     print('NO tem_files wrong_division_cells.pikcle, no dealing with CMap data')

    pd_number = pd.read_csv(config["number_dictionary"], names=["name", "label"])
    number_dict = pd.Series(pd_number.label.values, index=pd_number.name).to_dict()

    name_dict = {value: key for key, value in number_dict.items()}
    cell_tree = config["cell_tree"]

    df = read_new_cd(config['acetree_file'])
    df_t = df[df.time == time_point]
    nucleus_names = list(df_t.cell)  # all names based on nucleus location
    nucleus_number = [number_dict[cell_name] for cell_name in nucleus_names]

    ## extract nucleus location information in the aceTree
    nucleus_location = df_t.loc[:, ['z', 'x', 'y']].copy()
    ace_shape = config['image_size'].copy()
    # nucleus_location['z'] = nucleus_location['z'] * config['z_resolution'] / config['xy_resolution']
    # ace_shape[0] = ace_shape[0] * config['z_resolution'] / config['xy_resolution']
    nucleus_location = nucleus_location.values

    ## load seg volume
    # seg file is the "SegCellTimeCombined" FOlder
    seg = nib.load(seg_file).get_data().transpose([2, 1, 0]) # z, y , x
    config["res"] = ace_shape[-1] / seg.shape[-1] * config["xy_resolution"]
    nucleus_location_zoom = (nucleus_location * np.array(seg.shape) / np.array(ace_shape)).astype(np.uint16)
    nucleus_location_zoom[:, 0] = seg.shape[0] - nucleus_location_zoom[:, 0]
    # nucleus_location_zoom[:, 0] = seg.shape[0] - nucleus_location_zoom[:, 0]  # the embryo is reversed at z axis
    ####################To save nucleus location Data##########################
    nucleus_loc_to_save = pd.DataFrame.from_dict({'nucleus_label': nucleus_number, 'nucleus_name': nucleus_names,
                                                  f'x_{seg.shape[2]}': nucleus_location_zoom[:, 2],
                                                  f'y_{seg.shape[1]}': nucleus_location_zoom[:, 1],
                                                  f'z_{seg.shape[0]}': nucleus_location_zoom[:, 0]})
    save_name = os.path.join(config['save_nucleus_folder'], config['embryo_name'],
                             config['embryo_name'] + "_" + str(time_point).zfill(3) + '_nucLoc' + '.csv')
    save_name_fast_read = os.path.join(config["project_folder"], 'TemCellGraph', config['embryo_name'],
                                       config['embryo_name'] + "_" + str(time_point).zfill(3) + '_nucLoc' + '.txt')
    ##################################################
    #  unify the segmentation label
    # seg file is the "SegCellTimeCombined" FOlder
    unify_seg = np.zeros_like(seg)
    changed_flag = np.zeros_like(seg)  # to label whether a cell has been updated with labels in the nucleus stack.
    nucleus_loc_to_save["volume"] = ""  ################### Used for wrting cell information
    nucleus_loc_to_save["surface"] = ""  ################### Used for wrting cell information
    nucleus_loc_to_save["note"] = ""  ################### Used for wrting cell information
    for i, nucleus_loc in enumerate(list(nucleus_location_zoom)):
        target_label = nucleus_number[i]
        if "Nuc" in nucleus_names[i]:  # ignore all SegCell starting with "Nuc****"
            nucleus_loc_to_save.loc[nucleus_loc_to_save.nucleus_label == target_label, "note"] = "lost_hole"
            update_time_tree(config['embryo_name'], name_dict[target_label], time_point, file_lock, config, add=False)
            continue
        raw_label = seg[nucleus_loc[0], nucleus_loc[1], nucleus_loc[2]]
        update_flag = changed_flag[nucleus_loc[0], nucleus_loc[1], nucleus_loc[2]]
        # raw label is the label in SegCellTimeCOmbined
        if raw_label != 0:
            if not update_flag:
                config_this = {'mesh_path': config['mesh_path'], 'embryo_name': config['embryo_name'],
                               'time_point': str(config['time_point']).zfill(3), 'cell_label': target_label}
                # set the SegCellTimeCombined With CD file again, even SegCell is already unified actually
                unify_seg[seg == raw_label] = target_label
                changed_flag[seg == raw_label] = 1
                # add volume and surface information
                # surface_area = get_surface_area(seg == raw_label)
                volume,surface_area = get_volume_surface_area_adjusted_Alpha_Shape(config_this)
                nucleus_loc_to_save.loc[nucleus_loc_to_save.nucleus_label == target_label, "volume"] = volume * (config["res"] ** 3)
                nucleus_loc_to_save.loc[nucleus_loc_to_save.nucleus_label == target_label, "surface"] = surface_area * (config["res"] ** 2)
                # print('finish calculate volume adn surface')
            else:
                # check whether two labels from the same mother
                another_label = unify_seg[nucleus_loc[0], nucleus_loc[1], nucleus_loc[2]]
                another_mother_name = cell_tree.parent(name_dict[another_label]).tag
                mother_name = cell_tree.parent(name_dict[target_label]).tag
                if another_mother_name == mother_name:
                    mother_label = number_dict[mother_name]
                    ################### add a virtual meother nucleus to the nucleus loc file
                    mother_loc, ch1, ch2 = get_mother_loc(cell_tree, mother_name, nucleus_loc_to_save)
                    if mother_loc is not None:
                        unify_seg[seg == raw_label] = mother_label
                        nucleus_loc_to_save = nucleus_loc_to_save.append({
                            "nucleus_label": mother_label,
                            "nucleus_name": mother_name,
                            f'x_{seg.shape[2]}': mother_loc[0],
                            f'y_{seg.shape[1]}': mother_loc[1],
                            f'z_{seg.shape[0]}': mother_loc[2],
                            "note": "mother"
                        }, ignore_index=True)
                        # surface_area = get_surface_area(seg == raw_label)
                        config_this = {'mesh_path': config['mesh_path'], 'embryo_name': config['embryo_name'],
                                       'time_point': str(config['time_point']).zfill(3), 'cell_label': mother_label}
                        volume,surface_area = get_volume_surface_area_adjusted_Alpha_Shape(config_this)
                        nucleus_loc_to_save.loc[nucleus_loc_to_save.nucleus_label == mother_label, "volume"] = volume* (config["res"] ** 3)
                        nucleus_loc_to_save.loc[nucleus_loc_to_save.nucleus_label == mother_label, "surface"] = surface_area * (config["res"] ** 2)
                        # print('finish calculate volume adn surface')
                        # update daughter SegCell information
                        nucleus_loc_to_save = update_daughter_info(nucleus_loc_to_save, ch1, ch2, mother_name)
                        update_time_tree(config['embryo_name'], mother_name, time_point, file_lock, config, add=True)
                        update_time_tree(config['embryo_name'], ch1, time_point, file_lock, config, add=False)
                        update_time_tree(config['embryo_name'], ch2, time_point, file_lock, config, add=False)
                    else:
                        # The nucleus is also occupied
                        nucleus_loc_to_save.loc[
                            nucleus_loc_to_save.nucleus_label == target_label, "note"] = "lost_inner1"
                        update_time_tree(config['embryo_name'], name_dict[target_label], time_point, file_lock, config,
                                         add=False)
                else:
                    # nucleus is occupied by strangers
                    nucleus_loc_to_save.loc[nucleus_loc_to_save.nucleus_label == target_label, "note"] = "lost_inner2"
                    update_time_tree(config['embryo_name'], name_dict[target_label], time_point, file_lock, config,
                                     add=False)
        else:
            # nucleus locates in the background
            nucleus_loc_to_save.loc[nucleus_loc_to_save.nucleus_label == target_label, "note"] = "lost_hole"
            update_time_tree(config['embryo_name'], name_dict[target_label], time_point, file_lock, config, add=False)

    check_folder_exist(save_name)
    nucleus_loc_to_save.to_csv(save_name, index=False)  ######################
    check_folder_exist(save_name_fast_read)
    with open(save_name_fast_read, 'wb') as f:
        pickle.dump(nucleus_loc_to_save, f)

    ##  deal with dividing SegCell
    raw_labels = list(seg[nucleus_location_zoom[:, 0], nucleus_location_zoom[:, 1], nucleus_location_zoom[:, 2]])
    repeat_labels = [[i, label] for i, label in enumerate(raw_labels) if raw_labels.count(label) > 1]
    repeat_labels = [x for x in repeat_labels if x[1] != 0]
    # Label with 0 is the missed cell
    # reset all labels to their parent label
    division_seg = unify_seg.copy()
    cell_locations = [list(x) for x in list(nucleus_location_zoom)]
    cell_names = nucleus_names.copy()
    for repeat_label in repeat_labels:
        cell_name = name_dict[nucleus_number[repeat_label[0]]]
        cell_names.remove(cell_name)
        cell_locations.remove(list(nucleus_location_zoom[repeat_label[0]]))
        try:
            parent_node = cell_tree.parent(cell_name)
            parent_label = number_dict[parent_node.tag]
        except:
            pass

        # todo: danger usage 1- here
        # expect the two region condition:! please. brain is burning.
        # print([config['embryo_name'], parent_label, time_point])
        # print(list_wrong_division_label)
        # if [config['embryo_name'], name_dict[parent_label], time_point] in list_wrong_division_label:
        #     print([config['embryo_name'], name_dict[parent_label], time_point],'--match, wrong division - SegCellCombinedLabelUnified')
        #     continue

        division_seg[unify_seg == number_dict[cell_name]] = parent_label
        if name_dict[parent_label] not in cell_names:
            cell_names.append(name_dict[parent_label])
            cell_locations.append(list(nucleus_location_zoom[repeat_label[0]]))
    nuc_positions = dict(zip(cell_names, cell_locations))  # ABalappap

    return unify_seg, nuc_positions, config["res"]


def add_relation(point_graph, division_seg, name_dict, res,config_this):
    '''
    Add relationship information between SegCell. (contact surface area)
    :param point_graph: point graph of SegCell
    :param division_seg: cell segmentations
    :return point_graph: contact graph between cells
    '''
    if np.unique(division_seg).shape[0] > 2:  # in case there are multiple cells
        # contact_pairs, contact_area = get_contact_area(division_seg)
        contact_pairs, contact_area = get_contact_surface_adjusted_Alpha_Shape(config_this)
        for i, one_pair in enumerate(contact_pairs):
            point_graph.add_edge(name_dict[one_pair[0]], name_dict[one_pair[1]], area=contact_area[i] * (res ** 2))

    return point_graph


def get_contact_area_fast(volume):
    '''
    Get the contact volume surface of the segmentation. The segmentation results should be watershed segmentation results
    with a ***watershed line***.
    :param volume: segmentation result
    :return boundary_elements_uni: pairs of SegCell which contacts with each other
    :return contact_area: the contact surface area corresponding to that in the the boundary_elements.
    '''

    labels = np.unique(volume).tolist()
    labels.remove(0)

    contact_area = []
    boundary_elements_uni_new = []  # TODO: debug, some contacts are not detected
    for label1 in tqdm(labels):
        label1_mask = get_boundafry(volume == label1, b_width=2)
        label2s = get_contact_pairs(volume, label1, b_width=2)
        for label2 in label2s:
            if (label1, label2) in boundary_elements_uni_new or (label2, label1) in boundary_elements_uni_new:
                continue
            label2_mask = get_boundafry(volume == label2, b_width=2)
            contact_mask = np.logical_and(label1_mask, label2_mask)
            if contact_mask.sum() > 4:
                verts, faces, _, _ = marching_cubes_lewiner(contact_mask)
                area = mesh_surface_area(verts, faces) / 2
                contact_area.append(area)
                boundary_elements_uni_new.append((label1, label2))

    return boundary_elements_uni_new, contact_area


def get_contact_area(volume):
    '''
    Get the contact volume surface of the segmentation. The segmentation results should be watershed segmentation results
    with a ***watershed line***.
    :param volume: segmentation result
    :return boundary_elements_uni: pairs of SegCell which contacts with each other
    :return contact_area: the contact surface area corresponding to that in the the boundary_elements.
    '''

    cell_mask = volume != 0
    boundary_mask = get_boundafry(cell_mask)
    [x_bound, y_bound, z_bound] = np.nonzero(boundary_mask)
    boundary_elements = []
    for (x, y, z) in zip(x_bound, y_bound, z_bound):
        neighbors = volume[np.ix_(range(x - 1, x + 2), range(y - 1, y + 2), range(z - 1, z + 2))]
        neighbor_labels = list(np.unique(neighbors))
        neighbor_labels.remove(0)
        if len(neighbor_labels) == 2:
            boundary_elements.append(neighbor_labels)
    boundary_elements_uni = list(x.tolist() for x in np.unique(np.array(boundary_elements), axis=0))
    contact_area = []
    boundary_elements_uni_new = []
    for (label1, label2) in boundary_elements_uni:
        contact_mask = np.logical_and(ndimage.binary_dilation(volume == label1),
                                      ndimage.binary_dilation(volume == label2))
        contact_mask = np.logical_and(contact_mask, boundary_mask)
        if contact_mask.sum() > 4:
            verts, faces, _, _ = marching_cubes_lewiner(contact_mask)
            area = mesh_surface_area(verts, faces) / 2
            contact_area.append(area)
            boundary_elements_uni_new.append((label1, label2))
    return boundary_elements_uni_new, contact_area


def get_contact_surface_adjusted_Alpha_Shape(config_this):
    # print(config_this)
    mesh_path=config_this['mesh_path']
    embryo_name=config_this['embryo_name']
    time_point=config_this['time_point']
    cell_contact_pairs = []
    contact_area_list = []
    contact_saving_path = os.path.join(mesh_path, 'stat', embryo_name,embryo_name+'_'+time_point+'_segCell_contact.txt')
    try:
        with open(contact_saving_path,'rb') as handle:
            contact_dict = pickle.load(handle)
    except:
        print('open failed ',contact_saving_path)
        return cell_contact_pairs, contact_area_list

    # cell_mesh_dict={}
    for idx,contact_value in contact_dict.items():
        cell1=int(idx.split('_')[0])
        cell2=int(idx.split('_')[1])
        cell_contact_pairs.append((cell1,cell2))
        contact_area_list.append(contact_value)
    return cell_contact_pairs,contact_area_list

def construct_stat_embryo(cell_tree, max_time):
    '''
    Construct embryonic statistical DataFrom
    :param cell_tree: cell lineage tree used in the analysis
    :param max_time: the maximum time point we analyze.
    :return:
    '''
    global stat_embryo

    all_names = [cname for cname in cell_tree.expand_tree(mode=Tree.WIDTH)]
    # Get tuble lists with elements from the list
    name_combination = []
    first_level_names = []
    for i, name1 in enumerate(all_names):
        for name2 in all_names[i + 1:]:
            if not (cell_tree.is_ancestor(name1, name2) or cell_tree.is_ancestor(name2, name1)):
                first_level_names.append(name1)
                name_combination.append((name1, name2))

    multi_index = pd.MultiIndex.from_tuples(name_combination, names=['cell1', 'cell2'])
    stat_embryo = pd.DataFrame(np.full(shape=(max_time, len(name_combination)), fill_value=np.nan, dtype=np.float32),
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


def assemble_result(point_embryo, time_point, number_dict):
    '''
    Assemble results of the embryo at different time points into a single DataFrame in Pandas
    :param point_embryo: embryo information at one time point
    :param time_point: time point of the embryo
    :return st_embryo: statistical shape information. Checked through cell_name1, cell_name2, time.
    '''
    global stat_embryo
    edges_view = point_embryo.edges(data=True)
    # use saved graph to mark in the dataframe.csv
    for one_edge in edges_view:
        edge_weight = one_edge[2]['area']
        if (one_edge[0], one_edge[1]) in stat_embryo.columns:
            stat_embryo.loc[time_point, (one_edge[0], one_edge[1])] = edge_weight
        elif (one_edge[1], one_edge[0]) in stat_embryo.columns:
            stat_embryo.loc[time_point, (one_edge[1], one_edge[0])] = edge_weight
        else:
            pass

    # #   neighbors to text for GUI
    #     neighbors_file = os.path.join(config["para"]['seg_folder'], config["para"]['embryo_name']+ "_guiNieghbor", config["para"]['embryo_name'] + "_" +str(time_point).zfill(3)+"_guiNeighbor.txt")
    #     if not os.path.isdir(os.path.dirname(neighbors_file)):
    #         os.makedirs(os.path.dirname(neighbors_file))
    #     for i, cell_name in enumerate(point_embryo.nodes()):
    #         neighbor_cells = list(point_embryo.neighbors(cell_name))
    #         neighbor_labels = [str(number_dict[name]) for name in neighbor_cells]
    #         cell_label = str(number_dict[cell_name])
    #         label_str = ",".join(([cell_label] + neighbor_labels))  # first one master cell
    #         if i == 0:
    #             with open(neighbors_file, "w") as f:
    #                 f.write(label_str+"\n")
    #         else:
    #             with open(neighbors_file, "a") as f:
    #                 f.write(label_str+"\n")

    return stat_embryo


def get_mother_loc(cell_tree, mother_name, loc):
    '''
    Get mother nucleus location based on children's location
    :param cell_tree: cell nucleus lineage
    :param mother_name: mother cell name
    :param loc: daughter's location
    :return mother_loc: mother's location
    :return children_name1: first child's name
    :return children_name2: second child's name
    '''
    try:
        children1, children2 = cell_tree.children(mother_name)
        children_name1, children_name2 = [children1.tag, children2.tag]

        mother_loc = (loc[loc.nucleus_name == children_name1].iloc[:, 2:5].values[0] + \
                      loc[loc.nucleus_name == children_name2].iloc[:, 2:5].values[0]) / 2
    except:
        print("test here")
        return None, None, None

    return mother_loc.astype(np.int16).tolist(), children_name1, children_name2


def get_surface_area(cell_mask):
    '''
    get cell surface area
    :param cell_mask: single cell mask
    :return surface_are: cell's surface are
    '''
    # ball_structure = morphology.cube(3)
    # erased_mask = ndimage.binary_erosion(cell_mask, ball_structure, iterations=1)
    # surface_area = np.logical_and(~erased_mask, cell_mask).sum()
    verts, faces, _, _ = marching_cubes_lewiner(cell_mask)
    surface = mesh_surface_area(verts, faces)  # todo: didn't divided by 2????

    return surface


def get_volume_surface_area_adjusted_Alpha_Shape(config_this):
    # print(config_this)
    mesh_path=config_this['mesh_path']
    embryo_name=config_this['embryo_name']
    time_point=config_this['time_point']
    cell_label=config_this['cell_label']

    # if not found will set as 0. That's fine. THe post segmentation, shape analysis and assemble will work at PROJECT CellShapeAnalysis
    volume,surface=0.0,0.0
    volume_path = os.path.join(mesh_path, 'stat', embryo_name,embryo_name+'_'+time_point+'_segCell_volume.txt')
    surface_path = os.path.join(mesh_path, 'stat', embryo_name,embryo_name+'_'+time_point+'_segCell_surface.txt')
    # if cell_label ==486:
    #     with open(volume_path, 'rb') as handle:
    #         volume = pickle.load(handle)
    #     print(np.unique(volume))
    try:
        with open(volume_path, 'rb') as handle:
            volume = pickle.load(handle)[cell_label]
    except Exception as e:
        # print(e)
        print('open failed ', volume_path, cell_label)

    try:
        with open(surface_path, 'rb') as handle:
            surface = pickle.load(handle)[cell_label]
    except Exception as e:
        # print(e)
        print('open failed ', surface_path,cell_label)

    return volume, surface





def update_time_tree(embryo_name, cell_name, time_point, file_lock, config, add=False):
    '''
    Update cell lineage tree. Such as two nuclei in dividing cell are merged into one.
    :param embryo_name: embryo's name
    :param cell_name: cell's name
    :param time_point: time point of the embryo
    :param file_lock: file locker for parallel computing
    :param add: add or remove one cell in the lineage
    :return:
    '''
    file_lock.acquire()
    try:
        with open(os.path.join(config["stat_folder"], "{}_time_tree.txt".format(embryo_name)), 'rb') as f:
            time_tree = pickle.load(f)
        origin_times = time_tree.get_node(cell_name).data.get_time()
        if add:
            origin_times = origin_times + [time_point]
        else:
            origin_times.remove(time_point)
        time_tree.get_node(cell_name).data.set_time(origin_times)
        with open(os.path.join(config["stat_folder"], "{}_time_tree.txt".format(embryo_name)), 'wb') as f:
            pickle.dump(time_tree, f)
    except:
        pass
    finally:
        file_lock.release()
        # pass


def update_daughter_info(nucleus_loc_info, ch1, ch2, mother):
    '''
    add notes to the nucleus loc file.
    :param nucleus_loc_info: nucleus location including notes
    :param ch1: first child's name
    :param ch2: second child's name
    :param mother: mother's name
    :return nucleus_loc_info: updated nucleus info
    '''
    nucleus_loc_info.loc[nucleus_loc_info.nucleus_name == ch1, ["note", "volume", "surface"]] = [
        "child_of_{}".format(mother), '', '']
    nucleus_loc_info.loc[nucleus_loc_info.nucleus_name == ch2, ["note", "volume", "surface"]] = [
        "child_of_{}".format(mother), '', '']

    return nucleus_loc_info


def compose_surface_and_volume(embryo):
    loc_folder = "./output/NucleusLoc"
    # combien all volume and surface informace
    volume_stat = pd.DataFrame([], columns=[], dtype=np.float32)
    surface_stat = pd.DataFrame([], columns=[], dtype=np.float32)
    volume_lists = []
    surface_lists = []
    t_max = len(glob.glob(os.path.join(loc_folder, embryo, "*_nucLoc.csv")))
    for t in tqdm(range(1, t_max + 1), desc="Processing {}".format(embryo)):
        nucleus_loc_file = os.path.join(loc_folder, embryo,
                                        os.path.basename(embryo) + "_" + str(t).zfill(3) + "_nucLoc" + ".csv")
        pd_loc = pd.read_csv(nucleus_loc_file)
        cell_volume_surface = pd_loc[["nucleus_name", "volume", "surface"]]
        cell_volume_surface = cell_volume_surface.set_index("nucleus_name")
        volume_lists.append(cell_volume_surface["volume"].to_frame().T.dropna(axis=1))
        surface_lists.append(cell_volume_surface["surface"].to_frame().T.dropna(axis=1))
    volume_stat = pd.concat(volume_lists, keys=range(1, t_max + 1), ignore_index=True, axis=0, sort=False,
                            join="outer")
    surface_stat = pd.concat(surface_lists, keys=range(1, t_max + 1), ignore_index=True, axis=0, sort=False,
                             join="outer")
    volume_stat.index = list(range(1, t_max + 1))
    surface_stat.index = list(range(1, t_max + 1))
    volume_stat.to_csv(os.path.join("./statistics", embryo, embryo + "_volume" + '.csv'))
    surface_stat.to_csv(os.path.join("./statistics", embryo, embryo + "_surface" + '.csv'))




#  import dependent library
import os
import glob
import random
import numpy as np
from scipy import ndimage
from skimage.morphology import h_maxima, watershed, binary_opening
from scipy.spatial import Delaunay
from scipy.stats import mode

# import user defined library
from utils.data_io import nib_load, nib_save

#=========================================================================
#   main function for post process
#=========================================================================
def segment_membrane(para):
    embryo_name = para[0]
    file_name = para[1]
    egg_shell = para[2]
    name_embryo_T = "_".join(os.path.basename(file_name).split("_")[0:2])

    binary_embryo = nib_load(file_name) * egg_shell
    assert len(np.unique(binary_embryo)) == 2, "Post process can only process binary image"
    binary_cell = (binary_embryo == 0).astype(np.uint8)

    #  get local maximum graph
    point_list, edge_list, edge_weight_list = construct_weighted_graph(binary_cell, local_max_h=1)
    valid_edge_list = [edge_list[i] for i in range(len(edge_weight_list)) if edge_weight_list[i] < 10]
    #  combine background points
    point_tomerge_list = combine_background_maximum(valid_edge_list)
    merged_list = combine_inside_maximum(point_tomerge_list, point_tomerge_list)

    #  seeded watershed segmentation
    marker_volume = np.zeros_like(binary_embryo, dtype=np.uint8)
    marker_point_list = np.transpose(np.array(point_list), [1, 0]).tolist()
    marker_volume[marker_point_list[0], marker_point_list[1], marker_point_list[2]] = 1
    marker_volume = ndimage.morphology.binary_dilation(marker_volume, structure=np.ones((3, 3, 3), dtype=bool))
    marker_volume = ndimage.label(marker_volume)[0]  # each markers should be labeled with different values.
    memb_edt = ndimage.morphology.distance_transform_edt(binary_cell)
    memb_edt_reversed = memb_edt.max() - memb_edt
    watershed_seg = watershed(memb_edt_reversed, marker_volume.astype(np.uint16), watershed_line=True)

    #  reverse initial segmentation with previous merged point list.
    merged_seg = reverse_seg_with_max_cluster(watershed_seg, point_list, merged_list)
    #  set background label as zero
    background_label = mode(merged_seg, axis=None)[0][0]
    merged_seg[merged_seg == background_label] = 0
    merged_seg = set_boundary_zero(merged_seg)

    #  filter with nucleus stack
    nuc_seg = nib_load(os.path.join("./dataset/test", embryo_name, "SegNuc", name_embryo_T + "_segNuc.nii.gz"))
    cell_seg = cell_filter_with_nucleus(merged_seg, nuc_seg) #TODO: some nucleus are lost in the nucleus stack, so acetree is used to filter gaps when naming each segmented region

    # save result
    save_name = os.path.join("./output", embryo_name, "CellSeg", name_embryo_T+"_cellSeg.nii.gz")
    nib_save(cell_seg.astype(np.uint16), save_name)


#================================================================================
#       extra tools
#================================================================================
#  construct local maximum graph
def construct_weighted_graph(bin_image, local_max_h = 2):
    '''
    Construct edge weight graph from binary image.
    :param bin_image: cell binary image
    :return point_list: all points embedded in the triangulation, used for location query
    :return edge_list: list of edges in the triangulation
    :return edge_weight_list: edge weight corresponds to the edge list.
    '''

    volume_shape = bin_image.shape
    bin_cell = ndimage.morphology.binary_opening(bin_image).astype(np.float)
    bin_memb = bin_cell == 0
    bin_cell_edt = ndimage.morphology.distance_transform_edt(bin_cell)

    # get local maximum mask
    local_maxima_mask = h_maxima(bin_cell_edt, local_max_h)
    [maxima_x, maxima_y, maxima_z] = np.nonzero(local_maxima_mask)
    #  find boundary points to force large weight
    x0 = np.where(maxima_x == 0)[0]; x1 = np.where(maxima_x == volume_shape[0] - 1)[0]
    y0 = np.where(maxima_y == 0)[0]; y1 = np.where(maxima_y == volume_shape[1] - 1)[0]
    z0 = np.where(maxima_z == 0)[0]; z1 = np.where(maxima_z == volume_shape[2] - 1)[0]
    b_indx = np.concatenate((x0, y0, z0, x1, y1, z1), axis=None).tolist()


    point_list = np.stack((maxima_x, maxima_y, maxima_z), axis=1)
    tri_of_max = Delaunay(point_list)
    triangle_list = tri_of_max.simplices
    edge_list = []
    for i in range(triangle_list.shape[0]):
        for j in range(triangle_list.shape[1]-1):
            edge_list.append([triangle_list[i][j-1], triangle_list[i][j]])
        edge_list.append([triangle_list[i][j], triangle_list[i][0]])
    # add edges for all boundary points
    for i in range(len(b_indx)):
        for j in range(i, len(b_indx)):
            one_point = b_indx[i]
            another_point = b_indx[j]
            if ([one_point, another_point] in edge_list) or ([another_point, one_point] in edge_list):
                continue
            edge_list.append([one_point, another_point])

    weights_volume = bin_memb * 10000  # construct weights volume for graph
    edge_weight_list = []
    for one_edge in edge_list:
        start_x0 = point_list[one_edge[0]]
        end_x1   = point_list[one_edge[1]]
        if (one_edge[0] in b_indx) and (one_edge[1] in b_indx):
            edge_weight = 0  # All edges between boundary points are set as zero
        elif (one_edge[0] in b_indx) or (one_edge[1] in b_indx):
            edge_weight = 10000 * 10
        else:
            edge_weight = line_weight_integral(start_x0, end_x1, weights_volume)

        edge_weight_list.append(edge_weight)

    return point_list.tolist(), edge_list, edge_weight_list

#  integrate along a line
def line_weight_integral(x0, x1, weight_volume):

    # find all points between start and end
    inline_points = all_points_inline(x0, x1)
    points_num = inline_points.shape[0]
    line_weight = 0
    for i in range(points_num):
        point_weight = weight_volume[inline_points[i][0],
                                    inline_points[i][1],
                                    inline_points[i][2]]
        line_weight = line_weight + point_weight

    return line_weight

#  get all lines along a point
def all_points_inline(x0, x1):

    d = np.diff(np.array((x0, x1)), axis=0)[0]
    j = np.argmax(np.abs(d))
    D = d[j]
    aD = np.abs(D)
    return x0 + (np.outer(np.arange(aD + 1), d) + (aD >> 1)) // aD

    # backup points that should be labelled together
def combine_background_maximum(point_weight_list):
    point_tomerge_list0 = []
    for one_edge in point_weight_list:
        added_flag = 0
        point1 = one_edge[0]
        point2 = one_edge[1]
        for i in range(len(point_tomerge_list0)):
            if (point1 in point_tomerge_list0[i]) or (point2 in point_tomerge_list0[i]):
                point_tomerge_list0[i] = list(set().union([point1, point2], point_tomerge_list0[i]))
                added_flag = 1
                break
        if not added_flag:
            point_tomerge_list0.append([point1, point2])

    return point_tomerge_list0

    #  combine local maximums inside the embryo
def combine_inside_maximum(cluster1, cluster2):
    point_tomerge_list = []
    merged_cluster = []
    while len(cluster1):
        delete_index = []
        cluster_in1 = cluster1.pop()
        if cluster_in1 in merged_cluster:
            continue
        cluster_final = set(cluster_in1)
        for cluster_in2 in cluster2:
            tem_final = set(cluster_final).intersection(cluster_in2)
            if len(tem_final):
                merged_cluster.append(cluster_in2)
                cluster_final = set().union(cluster_final, cluster_in2)
        point_tomerge_list.append(list(cluster_final))
    return point_tomerge_list

#  reverse over-segmentation with merged maximum.
def reverse_seg_with_max_cluster(watershed_seg, init_list, max_cluster):
    merged_seg = watershed_seg.copy()
    for one_merged_points in max_cluster:
        first_point = init_list[one_merged_points[0]]
        one_label = watershed_seg[first_point[0], first_point[1], first_point[2]]
        for other_point in one_merged_points[1:]:
            point_location = init_list[other_point]
            new_label = watershed_seg[point_location[0], point_location[1], point_location[2]]
            merged_seg[watershed_seg == new_label] = one_label
        one_mask = merged_seg == one_label
        one_mask_closed = ndimage.binary_closing(one_mask)
        merged_seg[one_mask_closed!=0] = one_label

    return merged_seg

#  set all boundary torched labels as zero
def set_boundary_zero(pre_seg):
    '''
    SET_BOUNARY_ZERO is used to set all segmented regions attached to the boundary as zero background.
    :param pre_seg:
    :return:
    '''
    opened_mask = binary_opening(pre_seg)
    pre_seg[opened_mask==0] = 0
    seg_shape = pre_seg.shape
    boundary_mask = np.zeros_like(pre_seg, dtype=np.uint8)
    boundary_mask[0:2, :, :] = 1; boundary_mask[:, 0:2, :] = 1; boundary_mask[:, :, 0:2] = 1
    boundary_mask[seg_shape[0]-1:, :, :] = 1; boundary_mask[:, seg_shape[1]-1:, :] = 1; boundary_mask[:, :, seg_shape[2]-1:] = 1
    boundary_labels = np.unique(pre_seg[boundary_mask != 0])
    for boundary_label in boundary_labels:
        pre_seg[pre_seg == boundary_label] = 0

    return pre_seg

#   filter over-segmentation with binary nucleus loc
def cell_filter_with_nucleus(cell_seg, nuc_seg):
    init_all_labels = np.unique(cell_seg).tolist()
    keep_labels = (np.unique(cell_seg[nuc_seg > 0])).tolist()
    filtered_labels = list(set(init_all_labels) - set(keep_labels))
    for label in filtered_labels:
        cell_seg[cell_seg == label] = 0

    return cell_seg


def get_largest_connected_region(embryo_mask):
    label_structure = np.ones((3, 3, 3))
    embryo_mask = ndimage.morphology.binary_closing(embryo_mask, structure=label_structure)
    [labelled_regions, _]= ndimage.label(embryo_mask, label_structure)
    count_label = np.bincount(labelled_regions.flat)
    count_label[0] = 0
    valid_edt_mask0 = (labelled_regions == np.argmax(count_label))
    valid_edt_mask0 = ndimage.morphology.binary_closing(valid_edt_mask0)

    return valid_edt_mask0.astype(np.uint8)


#================================================================
#   get egg shell
#================================================================
def get_eggshell(wide_type_name, hollow=False):
    '''
    Get eggshell of specific embryo
    :param embryo_name:
    :return:
    '''
    wide_type_folder = os.path.join("./dataset/test", wide_type_name, "RawMemb")
    embryo_tp_list = glob.glob(os.path.join(wide_type_folder, "*.nii.gz"))
    random.shuffle(embryo_tp_list)
    overlap_num = 15 if len(embryo_tp_list) > 15 else len(embryo_tp_list)
    embryo_sum = nib_load(embryo_tp_list[0]).astype(np.float)
    for tp_file in embryo_tp_list[1:overlap_num]:
        embryo_sum += nib_load(tp_file)

    embryo_mask = otsu3d(embryo_sum)
    embryo_mask = get_largest_connected_region(embryo_mask)
    embryo_mask[0:2, :, :] = False; embryo_mask[:, 0:2, :] = False; embryo_mask[:, :, 0:2] = False
    embryo_mask[-3:, :, :] = False; embryo_mask[:, -3:, :] = False; embryo_mask[:, :, -3:] = False
    if hollow:
        dilated_mask = ndimage.morphology.binary_dilation(embryo_mask, np.ones((3,3,3)), iterations=2)
        eggshell_mask = np.logical_and(dilated_mask, ~embryo_mask)
        return eggshell_mask.astype(np.uint8)
    return embryo_mask.astype(np.uint8)


def otsu3d(gray):
    pixel_number = gray.shape[0] * gray.shape[1] * gray.shape[2]
    mean_weigth = 1.0/pixel_number
    his, bins = np.histogram(gray, np.arange(0,257))
    final_thresh = -1
    final_value = -1
    intensity_arr = np.arange(256)
    for t in bins[1:-1]: # This goes from 1 to 254 uint8 range (Pretty sure wont be those values)
        pcb = np.sum(his[:t])
        pcf = np.sum(his[t:])
        Wb = pcb * mean_weigth
        Wf = pcf * mean_weigth

        mub = np.sum(intensity_arr[:t]*his[:t]) / float(pcb)
        muf = np.sum(intensity_arr[t:]*his[t:]) / float(pcf)
        #print mub, muf
        value = Wb * Wf * (mub - muf) ** 2

        if value > final_value:
            final_thresh = t
            final_value = value
    final_img = gray.copy()
    final_img[gray > final_thresh] = 1
    final_img[gray < final_thresh] = 0
    return final_img
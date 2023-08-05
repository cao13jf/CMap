#!/usr/bin/env python
# -*- coding: utf-8 -*-

# import dependency library
import os
import math

import nibabel as nib
from nibabel.viewers import OrthoSlicer3D

import numpy as np
import pandas as pd

# import user defined library

def load_nitf2_img(path):
    return nib.load(path)


def show_nitf2_img(path):
    img = load_nitf2_img(path)
    OrthoSlicer3D(img.dataobj).show()
    return img


def deal_all(this_dir):
    for (root, dirs, files) in os.walk(this_dir):
        # print(files)
        for file in files:
            print(root, file)
            this_file_path = os.path.join(root, file)
            this_img = nib.load(this_file_path)
            print(this_img.shape)


# https://en.wikipedia.org/wiki/Spherical_coordinate_system

def descartes2spherical(points_xyz):
    """
    theta latitude range -pi/2 ~ pi/2
    phi longitude range 0~2*pi
    :param points_xyz:
    :return: [r,latitude(inclination angle,from y),longitude]
    """
    points_xyz = np.array(points_xyz)
    pts_sph = np.zeros(points_xyz.shape)

    # https://en.wikipedia.org/wiki/Spherical_coordinate_system
    # be careful, this is my transform base formula. There are many different transformation methods I thinks.
    xy = points_xyz[:, 0] ** 2 + points_xyz[:, 1] ** 2
    pts_sph[:, 0] = np.sqrt(xy + points_xyz[:, 2] ** 2)
    pts_sph[:, 1] = np.arctan2(np.sqrt(xy), points_xyz[:, 2]) - math.pi / 2  # lat phi
    # pts_sph[:,1] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    pts_sph[:, 2] = np.arctan2(points_xyz[:, 1], points_xyz[:, 0]) % (2 * math.pi)  # lon theta
    return pts_sph


def sph2descartes(points_sph):
    """

    :param points_sph: latitude is used in latitude -90-90 degrees
    :return:
    """
    points_sph = np.array(points_sph)
    points_xyz = np.zeros(points_sph.shape)

    radius = points_sph[:, 0]
    lat = points_sph[:, 1] - math.pi / 2
    lon = points_sph[:, 2]

    # https://en.wikipedia.org/wiki/Spherical_coordinate_system
    # be careful, this is my transform base formula. There are many different transformation methods I thinks.
    points_xyz[:, 0] = np.cos(lon) * np.sin(lat) * radius
    points_xyz[:, 1] = np.sin(lon) * np.sin(lat) * radius
    points_xyz[:, 2] = np.cos(lat) * radius

    return -points_xyz


# the code reference: https://www.thinbug.com/q/4116658
# https://en.wikipedia.org/wiki/Spherical_coordinate_system
def descartes2spherical2(points_xyz):
    """
    theta latitude range 0~pi
    phi longitude range 0~2*pi

    :param points_xyz:
    :return: [r,co-latitude(from z),longitude]
    """
    points_xyz = np.array(points_xyz)
    pts_sph = np.zeros(points_xyz.shape)

    # https://en.wikipedia.org/wiki/Spherical_coordinate_system
    # be careful, this is my transform base formula. There are many different transformation methods I thinks.
    xy = points_xyz[:, 0] ** 2 + points_xyz[:, 1] ** 2
    pts_sph[:, 0] = np.sqrt(xy + points_xyz[:, 2] ** 2)
    pts_sph[:, 1] = np.arctan2(np.sqrt(xy), points_xyz[:, 2])  # colat phi
    # pts_sph[:,1] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    pts_sph[:, 2] = np.arctan2(points_xyz[:, 1], points_xyz[:, 0]) % (2 * math.pi)  # lon theta
    return pts_sph


def sph2descartes2(points_sph):
    """

    :param points_sph: used in co-latitude 0-180
    :return:
    """
    points_sph = np.array(points_sph)
    points_xyz = np.zeros(points_sph.shape)

    radius = points_sph[:, 0]
    lat = points_sph[:, 1] % (math.pi)
    lon = points_sph[:, 2]

    # https://en.wikipedia.org/wiki/Spherical_coordinate_system
    # be careful, this is my transform base formula. There are many different transformation methods I thinks.
    points_xyz[:, 0] = np.cos(lon) * np.sin(lat) * radius
    points_xyz[:, 1] = np.sin(lon) * np.sin(lat) * radius
    points_xyz[:, 2] = np.cos(lat) * radius

    return points_xyz


def to_the_power_expand(modified_array):
    """

    :param modified_array:
    :return:
    """
    rows = modified_array.shape[0]
    columns = modified_array.shape[1]
    return_array = np.zeros(modified_array.shape)
    for i in range(rows):
        for j in range(columns):
            if modified_array[i][j] > 0:
                return_array[i][j] = (modified_array[i][j] / 10) ** 2
            elif modified_array[i][j] < 0:
                return_array[i][j] = -(-modified_array[i][j] / 10) ** 2
            else:
                return_array[i][j] = 0

    return return_array


def sqrt_expand(original_array):
    """

    :param original_array:
    :return:
    """
    rows = original_array.shape[0]
    columns = original_array.shape[1]
    return_array = np.zeros(original_array.shape)
    for i in range(rows):
        for j in range(columns):
            if original_array[i][j] > 0:
                return_array[i][j] = np.sqrt(original_array[i][j]) * 10
            elif original_array[i][j] < 0:
                return_array[i][j] = -np.sqrt(-original_array[i][j]) * 10
            else:
                return_array[i][j] = 0

    return return_array


def log_expand_offset(original_array, offset):
    """
    many problem!!!!!!!!!!!!!!!!!!!!!!!!!!!
    offset plus and minus is very bad for log
    in math , it is not good.
    :param original_array:
    :param offset:
    :return:
    """
    rows = original_array.shape[0]
    columns = original_array.shape[1]
    return_array = np.zeros(original_array.shape)
    for i in range(rows):
        for j in range(columns):
            if original_array[i][j] > 0:
                return_array[i][j] = math.log(original_array[i][j]) + offset
            elif original_array[i][j] < 0:
                return_array[i][j] = -math.log(-original_array[i][j]) - offset
            else:
                return_array[i][j] = 0

    return return_array


def exp_expand_offset(modified_array, offset):
    rows = modified_array.shape[0]
    columns = modified_array.shape[1]
    return_array = np.zeros(modified_array.shape)
    for i in range(rows):
        for j in range(columns):
            if modified_array[i][j] > 0:
                return_array[i][j] = math.exp(modified_array[i][j] - offset)
            elif modified_array[i][j] < 0:
                return_array[i][j] = -math.exp(-(modified_array[i][j] + offset))
            else:
                return_array[i][j] = 0

    return return_array


def read_csv_to_df(csv_path):
    """

    :param csv_path:
    :return:
    """
    # ---------------read SHcPCA coefficient-------------------
    # if not os.path.exists(csv_path):
    #     print('error detected! no SHcPCA matrix csv file can be found')
    #     return
    df_read = pd.read_csv(csv_path)
    df_index_tmp = df_read.values[:, :1]
    df_read.drop(columns=df_read.columns[0], inplace=True)
    df_read.index = list(df_index_tmp.flatten())
    return df_read
    # -----------------------------------------------------------------------


def combine_all_embryo_SHc_in_df(dir_my_data_SH_time_domain_csv, l_degree=25, is_norm=True):
    """
    combine 17 embryo spherical harmonic feature vector to one, just for ease of reading
    :param dir_my_data_SH_time_domain_csv:
    :param l_degree:
    :param is_norm:
    :return:
    """
    embryo_name_tmp = 'Sample{}LabelUnified'.format(f'{4:02}')
    if is_norm:
        path_saving_csv_normalized_tmp = os.path.join(dir_my_data_SH_time_domain_csv,
                                                      embryo_name_tmp + '_l_' + str(l_degree) + '_norm.csv')
    else:
        path_saving_csv_normalized_tmp = os.path.join(dir_my_data_SH_time_domain_csv,
                                                      embryo_name_tmp + '_l_' + str(l_degree) + '.csv')
    together_df = pd.DataFrame(columns=read_csv_to_df(path_saving_csv_normalized_tmp).columns)

    for cell_index in np.arange(start=4, stop=21, step=1):
        # path_tmp = r'./DATA/SegmentCellUnified04-20/Sample' + f'{cell_index:02}' + 'LabelUnified'
        # print(path_tmp)
        # ===========================draw lineage for one embryo=======================================================

        embryo_num = f'{cell_index:02}'

        embryo_name = 'Sample{}LabelUnified'.format(embryo_num)
        print(embryo_name)

        if is_norm:
            path_saving_csv_normalized = os.path.join(dir_my_data_SH_time_domain_csv,
                                                      embryo_name + '_l_' + str(l_degree) + '_norm.csv')
        else:
            path_saving_csv_normalized = os.path.join(dir_my_data_SH_time_domain_csv,
                                                      embryo_name + '_l_' + str(l_degree) + '.csv')
        norm_df = read_csv_to_df(path_saving_csv_normalized)
        # go through this norm df
        print('reconstructing the index in together dataframe')
        for item in norm_df.index:
            # print(item)
            together_df.loc[embryo_num + '::' + item] = norm_df.loc[item]
        print(together_df)
    if is_norm:

        together_df.to_csv(os.path.join(dir_my_data_SH_time_domain_csv, 'SHc_norm.csv'))
    else:
        together_df.to_csv(os.path.join(dir_my_data_SH_time_domain_csv, 'SHc.csv'))


def rotate_points_lon(points_list, phi):
    '''
    :param points_list:
    :param phi: radius 0-2pi
    :return:
    '''
    # theta range 0-pi
    # phi range 0-2*pi
    sph_points_original = descartes2spherical(points_list)
    sph_points_original[:, 2] = sph_points_original[:, 2] + phi
    return sph2descartes(sph_points_original)


def rotate_points_lat(points_list, theta):
    '''
    :param points_list:
    :param theta: radius -pi/2-pi/2
    :return:
    '''
    # theta range 0-pi
    # phi range 0-2*pi
    # theta = -theta
    rotation_matrix = [[1, 0, 0], [0, math.cos(theta), -math.sin(theta)], [0, math.sin(theta), math.cos(theta)]]
    return np.array(points_list).dot(np.array(rotation_matrix))

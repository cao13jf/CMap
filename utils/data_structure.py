'''
This llibrary defines all structures that will be used in the shape analysis
'''

import os
import glob
import pickle
from termcolor import colored
import pandas as pd
from treelib import Tree, Node

def read_new_cd(cd_file):
    df_nuc = pd.read_csv(cd_file, lineterminator="\n")
    df_nuc[["cell", "time"]] = df_nuc["Cell & Time"].str.split(":", expand=True)
    df_nuc = df_nuc.rename(columns={"X (Pixel)":"x", "Y (Pixel)":"y", "Z (Pixel)\r":"z"})
    df_nuc = df_nuc.astype({"x":float, "y":float, "z":float, "time":int})

    return df_nuc

def construct_celltree(nucleus_file, max_time):
    '''
    Construct cell tree structure with cell names
    :param nucleus_file:  the name list file to the tree initilization
    :param max_time: the maximum time point to be considered
    :return cell_tree: cell tree structure where each time corresponds to one cell (with specific name)
    '''

    ##  Construct cell
    #  Add unregulized naming
    cell_tree = Tree()
    cell_tree.create_node('P0', 'P0')
    cell_tree.create_node('AB', 'AB', parent='P0')
    cell_tree.create_node('P1', 'P1', parent='P0')
    cell_tree.create_node('EMS', 'EMS', parent='P1')
    cell_tree.create_node('P2', 'P2', parent='P1')
    cell_tree.create_node('P3', 'P3', parent='P2')
    cell_tree.create_node('C', 'C', parent='P2')
    cell_tree.create_node('P4', 'P4', parent='P3')
    cell_tree.create_node('D', 'D', parent='P3')
    cell_tree.create_node('Z2', 'Z2', parent='P4')
    cell_tree.create_node('Z3', 'Z3', parent='P4')

    # EMS
    cell_tree.create_node('E', 'E', parent='EMS')
    cell_tree.create_node('MS', 'MS', parent='EMS')

    # Read the name excel and construct the tree with complete segCell
    df_time = read_new_cd(nucleus_file)

    # read and combine all names from different acetrees
    ## Get cell number
    try:
        pd_number = pd.read_csv('./dataset/number_dictionary.csv', names=["name", "label"])
        number_dictionary = pd.Series(pd_number.label.values, index=pd_number.name).to_dict()
    except:
        raise Exception("Not find number dictionary at ./dataset")

    # =====================================
    # dynamic update the name dictionary
    # =====================================
    cell_in_dictionary = list(number_dictionary.keys())

    ace_pd = read_new_cd(os.path.join(nucleus_file))
    ace_pd = ace_pd[ace_pd.time <= max_time]
    cell_list = list(ace_pd.cell.unique())
    add_cell_list = list(set(cell_list) - set(cell_in_dictionary))

    if len(add_cell_list) != 0:
        assert len(add_cell_list) == 0, "Name dictionary should be updated"
        print(colored("Name dictionary updated", "red"))

        # ================================= cancel dynamic updating ============
        # add_cell_list.sort()
        # print("Name dictionary updated !!!")
        # add_number_dictionary = dict(zip(add_cell_list, range(len(cell_in_dictionary) + 1, len(cell_in_dictionary) + len(add_cell_list) + 1)))
        # number_dictionary.update(add_number_dictionary)
        # pd_number_dictionary = pd.DataFrame.from_dict(number_dictionary, orient="index")
        # pd_number_dictionary.to_csv('./dataset/number_dictionary.csv', header=False)

    df_time = df_time[df_time.time <= max_time]
    all_cell_names = list(df_time.cell.unique())
    for cell_name in list(all_cell_names):
        if cell_name not in number_dictionary:
            continue
        times = list(df_time.time[df_time.cell==cell_name])
        cell_info = cell_node()
        cell_info.set_number(number_dictionary[cell_name])
        cell_info.set_time(times)
        if not cell_tree.contains(cell_name):
            if "Nuc" not in cell_name:
                parent_name = cell_name[:-1]
                cell_tree.create_node(cell_name, cell_name, parent=parent_name, data=cell_info)
        else:
            cell_tree.update_node(cell_name, data=cell_info)

    return cell_tree, max_time


class cell_node(object):
    # Node Data in cell tree
    def __init__(self):
        self.number = 0
        self.time = 0

    def set_number(self, number):
        self.number = number

    def get_number(self):

        return self.number

    def set_time(self, time):
        self.time = time

    def get_time(self):

        return self.time
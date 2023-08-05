'''
This llibrary defines all structures that will be used in the shape analysis
'''

import os
import glob
import pandas as pd
from treelib import Tree

from .data_structure import read_cd_file

def add_number_dict(nucleus_file, max_time):
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

    # # Read the name excel and construct the tree with complete segCell
    # df_time = read_cd_file(nuc_file)

    # read and combine all names from different acetrees
    ## Get cell number
    try:
        label_name_dict = pd.read_csv(r'./dataset/name_dictionary.csv', index_col=0).to_dict()['0']
        name_label_dict = {value: key for key, value in label_name_dict.items()}

    except:
        name_label_dict = {}

    # =====================================
    # dynamic update the name dictionary
    # =====================================
    cell_in_dictionary = list(name_label_dict.keys())

    ace_pd = read_cd_file(os.path.join(nucleus_file))

    ace_pd = ace_pd[ace_pd.time <= max_time]
    cell_list = list(ace_pd.cell.unique())
    print('Old name list length:',len(cell_in_dictionary) ,'New name list length', len(cell_list))
    add_cell_list = list(set(cell_list) - set(cell_in_dictionary))
    add_cell_list.sort()
    print('Updated cells:',add_cell_list)

    if len(add_cell_list) > 0:
        print("Name dictionary updated !!!")
        add_number_dictionary = dict(zip(add_cell_list, range(len(cell_in_dictionary) + 1, len(cell_in_dictionary) + len(add_cell_list) + 1)))
        # --------save name_label_dict csv-------------
        name_label_dict.update(add_number_dictionary)
        pd_number_dictionary = pd.DataFrame.from_dict(name_label_dict, orient="index")
        print(pd_number_dictionary)
        pd_number_dictionary.to_csv('./dataset/number_dictionary.csv')

        # -----------save label_name_dict csv
        label_name_dict_saving={value: key for key, value in name_label_dict.items()}
        pd_name_dictionary = pd.DataFrame.from_dict(label_name_dict_saving, orient="index")
        print(pd_name_dictionary)
        pd_name_dictionary.to_csv('./dataset/name_dictionary.csv')


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



if __name__ == "__main__":

    CD_folder = r"./necessary_files/CD_files"
    nuc_files = sorted(glob.glob(os.path.join(CD_folder, "*.csv")))

    for idx, nuc_file in enumerate(nuc_files):
        add_number_dict(nuc_file, max_time=1000)

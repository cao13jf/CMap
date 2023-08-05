import os
import pickle
from tqdm import tqdm
import pandas as pd

from utils.data_structure import construct_celltree
from utils.stat_tools import generate_name_series

def generate_qc(args):
    embryo_names = args.test_embryos
    max_times=args.max_times
    volume_folder = "./statistics"
    save_folder = "./statistics"

    overall_threshold = 0.1

    # === get name dictionary
    pd_number = pd.read_csv("./dataset/number_dictionary.csv", names=["name", "label"])
    number_dict = pd.Series(pd_number.label.values, index=pd_number.name).to_dict()

    # =====================================================
    # Detect error based on volume consistency
    # =====================================================
    for idx,embryo_name in enumerate(tqdm(embryo_names, desc="QC: generating quality control files based on cell volume")):
        # time_tree_file = os.path.join(volume_folder, embryo_name, embryo_name+"_time_tree.txt")
        volume_file = os.path.join(volume_folder, embryo_name, embryo_name+"_volume.csv")
        volume_pd = pd.read_csv(volume_file, index_col=0, header=0)
        volume_pd.index = list(range(1, len(volume_pd) + 1, 1))
        cell_names = list(volume_pd)

        CMap_data_path = r'./'
        cd_file_path = os.path.join(CMap_data_path, 'dataset/test', embryo_name, 'CD{}.csv'.format(embryo_name))
        cell_tree,_=construct_celltree(cd_file_path,max_time=max_times[idx])


        # ==================================================
        # 1. Get the degree of cell volume diverge
        # ==================================================
        volume_pd_mean = volume_pd.mean(axis=0, skipna=True)
        volume_pd_median = volume_pd.median(axis=0, skipna=True)
        volume_pd_dif = (volume_pd - volume_pd_median).abs()
        volume_pd_diverge = volume_pd_dif.divide(volume_pd_median)

        tem = volume_pd_diverge.stack()

        outer_number = len(tem.index) # int(len(tem.index) * overall_threshold / 3)

        outer_diverge = volume_pd_diverge.stack().nlargest(outer_number, keep="first").rename("Volume Divergence of Cell")
        outer_names = generate_name_series(number_dict, outer_diverge)
        outer_diverge = pd.concat([outer_names, outer_diverge], axis=1, sort=True)


        # ==================================================
        #  2. The volume of two descendents should be constrained by their parent
        # ==================================================
        child2parent_ratio_dict = [] # {"Cell Name": volume of its parent cell}
        names = []
        for cell_name in cell_names:
            if not cell_tree[cell_name].is_leaf():
                children = cell_tree.children(cell_name)
                children = [child.tag for child in children]
                if (children[0] in cell_names) and (children[1] in cell_names):
                    children_volume = volume_pd[children[0]] + volume_pd[children[1]]
                    ratio = abs(children_volume - volume_pd_median[cell_name]) / volume_pd_median[cell_name]
                    child2parent_ratio_dict += [ratio, ratio]
                    names += children

        child2parent_volume = pd.concat(child2parent_ratio_dict, axis=1)
        child2parent_volume.columns = names

        outer_child2parent = child2parent_volume.stack().nlargest(outer_number, keep="first").rename("Children Volume Difference")
        outer_names = generate_name_series(number_dict, outer_child2parent)
        outer_child2parent = pd.concat([outer_names, outer_child2parent], axis=1, sort=True)


        # ==================================================
        # 3. Check whether one cell is monotonically changing
        # ==================================================
        raw_volume_diff = volume_pd.diff(axis=0).abs()
        raw_volume_diff = raw_volume_diff.drop([1])
        raw_volume_diff.index = list(range(1, len(raw_volume_diff)+1, 1))
        diff2median_ratio = raw_volume_diff.divide(volume_pd_median)

        outer_diff = diff2median_ratio.stack().nlargest(outer_number, keep="first").rename("Volume Difference to Previous TP")
        outer_names = generate_name_series(number_dict, outer_diff)
        outer_diff = pd.concat([outer_names, outer_diff], axis=1, sort=True)

        outer_all = outer_diverge.combine_first(outer_child2parent).combine_first(outer_diff)
        label_column = outer_all.pop("Label").astype("int16")
        outer_all.insert(0, "Label", label_column)
        outer_all.to_csv(os.path.join(save_folder, embryo_name, embryo_name+"_qc.csv"), index_label=["Time Point", "Cell Name"])
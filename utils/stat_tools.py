import pandas as pd


# ====================================================
# Generate total number of cells in each TP
# ====================================================
def generate_name_series(name_dict, outer):
    out_index = outer.index
    label_dict = {(tp, cell_name): name_dict[cell_name] for tp, cell_name in out_index}

    return pd.Series(label_dict, name="Label")
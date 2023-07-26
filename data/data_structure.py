'''
This llibrary defines all structures that will be used in the shape analysis
'''


import pandas as pd




def read_cd_file(cd_file_path):
    df_nuc_tmp = pd.read_csv(cd_file_path)
    df_nuc=df_nuc_tmp[['cell','time','x','y','z']]
    df_nuc = df_nuc.astype({"x": float, "y": float, "z": float, "time": int})
    return df_nuc

'''testing trained model'''
#  import dependency library
import os
import time
import glob
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
import nibabel as nib
import torch.nn.functional as F
from skimage.transform import resize

#  import user defined library
from utils.data_io import nib_save, img_save
from utils.ProcessLib import segment_membrane, get_largest_connected_region, get_eggshell

def validate(valid_loader, model, savepath=None, names=None, scoring=False, verbose=False, save_format=".nii.gz",
             snapsot=False, postprocess=False):
    H, W, T = 205, 285, 134  # input size to the network
    model.eval()
    runtimes = []
    for i, data in enumerate(tqdm(valid_loader, desc="Getting binary membrane:")):
        if scoring:
            target_cpu = data["seg_memb"][0, 0, :, :, :].numpy() if scoring else None
            x, target = data["raw_memb"], data["seg_memb"]  # TODO: batch = in prediction
        else:
            x = data[0]  # TODO: change collate in dataloader
        #  go through the network
        start_time = time.time()
        log_predict = model(x)   # [1, 2, depth, width, height]
        elapsed_time = time.time() - start_time
        runtimes.append(elapsed_time)
        prediction = F.softmax(log_predict, dim=1).squeeze()
        #  get measurement
        prediction = prediction.cpu().numpy()
        prediction = prediction.argmax(0).transpose([1, 2, 0])  # [channel, height, width, depth]
        prediction = resize(prediction.astype(np.float), (H, W, T), mode='constant', cval=0, order=0, anti_aliasing=True).astype(np.uint8)

        #  post process
        if postprocess == True:
            pass  # TODO: add postprocess

        #  save volume and snapshot data
        # prediction = get_largest_connected_region(prediction)
        if savepath is not None:
            if "npy" in save_format.lower():
                np.save(os.path.join(savepath,  names[i].split("_")[0], "MembBin", names[i] + "_membBin"), prediction)
            elif "nii.gz" in save_format.lower():
                save_name = os.path.join(savepath, names[i].split("_")[0],  "MembBin",  names[i] + "_membBin.nii.gz")
                nib_save(prediction, save_name)
                #  save snapshot
                if snapsot:
                    file_name = os.path.join(savepath, "snapshot", names[i] + str(60) + ".png")
                    img_save(prediction[:, :, 60], file_name)

def membrane2cell(args):
        for embryo_name in args.test_embryos:
            embryo_mask = get_eggshell(embryo_name)

            file_names = glob.glob(os.path.join("./output",embryo_name, "MembBin",'*.nii.gz'))
            parameters = []
            for file_name in file_names:
                parameters.append([embryo_name, file_name, embryo_mask])
                # segment_membrane(parameters)  # test without parallel compu
        mpPool = mp.Pool(mp.cpu_count() - 1)
        for _ in tqdm(mpPool.imap_unordered(segment_membrane, parameters), total=len(parameters), desc="membrane --> cell"):
            pass


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
             snapsot=None, postprocess=False):
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
        pred_dis, pred_bin = model(x)   # [1, 2, depth, width, height]
        elapsed_time = time.time() - start_time
        runtimes.append(elapsed_time)
        #  Regression only has one channel
        if pred_bin.shape[1] > 1:
            # pred_bin = F.softmax(pred_bin, dim=1)
            pred_bin = pred_bin.argmax(1) # [channel, height, width, depth]

        #  binary prediction
        pred_bin = pred_bin.cpu().numpy()
        pred_bin = pred_bin.squeeze().transpose([1, 2, 0])
        pred_bin = resize(pred_bin.astype(np.float), (H, W, T), mode='constant', cval=0, order=0, anti_aliasing=True)
        #  distance prediction
        pred_dis = pred_dis.cpu().numpy()
        pred_dis = pred_dis.squeeze().transpose([1, 2, 0])
        pred_dis = resize(pred_dis.astype(np.float), (H, W, T), mode='constant', cval=0, order=0, anti_aliasing=True)

        #  post process
        if postprocess == True:
            pass  # TODO: add postprocess

        #  save volume and snapshot data
        # prediction = get_largest_connected_region(prediction)
        if snapsot is not None:
            time.sleep(2)
            x =  x.cpu().numpy()
            x = x.transpose([0, 1, 3, 4, 2])
            image_dict = dict(Raw=x[0, 0, :, 100, :], bin=pred_bin[:, 100, :], dis=pred_dis[:, 100, :])
            snapsot.show_current_images(image_dict)
        if savepath is not None:
            if "npy" in save_format.lower():
                np.save(os.path.join(savepath,  names[i].split("_")[0], "MembBin", names[i] + "_membBin"), pred_bin)
            elif "nii.gz" in save_format.lower():
                save_name = os.path.join(savepath, names[i].split("_")[0],  "MembBin",  names[i] + "_membBin.nii.gz")
                nib_save(pred_bin.astype(np.uint8), save_name)

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


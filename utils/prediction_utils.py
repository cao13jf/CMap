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
        prediction = model(x)   # [1, 2, depth, width, height]
        elapsed_time = time.time() - start_time
        runtimes.append(elapsed_time)
        #  Regression only has one channel
        if prediction.shape[1] > 1:
            prediction = F.softmax(prediction, dim=1)
            prediction = prediction.argmax(0) # [channel, height, width, depth]

        prediction = prediction.cpu().numpy()
        prediction = prediction.squeeze().transpose([1, 2, 0])
        prediction = resize(prediction.astype(np.float), (H, W, T), mode='constant', cval=0, order=0, anti_aliasing=True)

        #  post process
        if postprocess == True:
            pass  # TODO: add postprocess

        #  save volume and snapshot data
        # prediction = get_largest_connected_region(prediction)
        if snapsot is not None:
            time.sleep(2)
            image_dict = dict(Raw=x[0, 0, :, :, 60], Prediction=prediction[120, :, :])
            snapsot.show_current_images(image_dict)
        if savepath is not None:
            if "npy" in save_format.lower():
                np.save(os.path.join(savepath,  names[i].split("_")[0], "MembBin", names[i] + "_membBin"), prediction)
            elif "nii.gz" in save_format.lower():
                save_name = os.path.join(savepath, names[i].split("_")[0],  "MembBin",  names[i] + "_membBin.nii.gz")
                nib_save((prediction > 0.9).astype(np.uint8), save_name)

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


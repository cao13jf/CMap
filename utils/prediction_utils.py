'''testing trained model'''
#  import dependency library
import os
import time
import numpy as np
from tqdm import tqdm
import nibabel as nib
import torch.nn.functional as F
from skimage.transform import resize

#  import user defined library
from utils.data_io import nib_save, img_save

def validate(valid_loader, model, savepath=None, names=None, scoring=False, verbose=False, save_format=".nii.gz",
             snapsot=False, postprocess=False):
    H, W, T = 205, 285, 134  # input size to the network
    model.eval()
    runtimes = []
    for i, data in enumerate(tqdm(valid_loader, desc="Segmenting testing data:")):
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
        prediction = resize(prediction, (H, W, T), mode='constant', cval=0, order=0, anti_aliasing=True)

        #  post process
        if postprocess == True:
            pass  # TODO: add postprocess

        #  save volume and snapshot data
        if savepath is not None:
            if "npy" in save_format.lower():
                np.save(os.path.join(savepath, names[i].split("_")[0], names[i] + "_predtion"), prediction)
            elif "nii.gz" in save_format.lower():
                save_name = os.path.join(savepath, names[i].split("_")[0], names[i] + "_prediction.nii.gz")
                nib_save(prediction, save_name)
                #  save snapshot
                if snapsot:
                    file_name = os.path.join(savepath, "snapshot", names[i] + str(60) + ".png")
                    img_save(prediction[:, :, 60], file_name)
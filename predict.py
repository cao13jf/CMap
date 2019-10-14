'''testing trained model'''
#  import dependency library
import os
import time
import imageio
import numpy as np
from tqdm import tqdm
import nibabel as nib
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

#  import user defined library
from data.transforms import Resize, Pad
from utils.criterions import dice_score, softmax_dice_score

cudnn.benchmark = True
path = os.path.dirname(__file__)


def validate(valid_loader, model, cfg, savepath=None, names=None, scoring=False, verbose=False, save_format=".nii.gz",
             snapsot=False, postprocess=False):
    H, W, T = 205, 285, 134  # input size to the network
    model.eval()
    runtimes = []
    for i, data in tqdm(enumerate(valid_loader), desc="Segmenting testing data:"):
        if scoring:
            target_cpu = data[1][0, :, :, :].numpy() if scoring else None
            x, target = data[:2]  # TODO: batch = in prediction
        else:
            x = data[0][np.newaxis, ...]  # TODO: change collate in dataloader
        #  go through the network
        start_time = time.time()
        log_predict = model()
        elapsed_time = time.time() - start_time
        runtimes.append(elapsed_time)
        prediction = F.softmax(log_predict, dim=1)
        #  get measurement
        prediction = prediction.cpu().numpy()
        prediction = prediction.argmax(0).transpose([0, 1])  # [channel, height, width, depth]

        #  post process
        if postprocess == True:
            pass  # TODO: add postprocess

        #  save volume and snapshot data
        if savepath is not None:
            if "npy" in save_format.lower():
                np.save(os.path.join(savepath, names[i] + "_predtion"), prediction)
            elif "nii.gz" in save_format.lower():
                save_name = os.path.join(savepath, names[i] + "_prediction.nii.gz")
                seg_img = prediction  # TODO: recover the volume to original size
                nib.save(nib.Nifti1Image(seg_img, None), save_name)
                #  save snapshot
                if snapsot:
                    snapshot_img = seg_img[:, :, 60]
                    os.makedirs(os.path.join(savepath, "snapshot", names[i]), exist_ok=True)
                    imageio.imwrite(os.path.join(savepath, names[i]+str(60)+".png"), snapshot_img)
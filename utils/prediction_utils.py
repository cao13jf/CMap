'''testing trained model'''
#  import dependency library
import os
import time
import glob
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
from skimage.transform import resize

#  import user defined library
from utils.data_io import nib_save
from utils.ProcessLib import segment_membrane, get_eggshell, combine_division, delete_isolate_labels

def validate(valid_loader, model, savepath=None, names=None, scoring=False, verbose=False, save_format=".nii.gz",
             snapsot=None, postprocess=False, size=None):
    model.eval()
    runtimes = []
    for i, data in enumerate(tqdm(valid_loader, desc="Getting binary membrane:")):
        if scoring:
            target_cpu = data["seg_memb"][0, 0, :, :, :].numpy() if scoring else None
            x, target = data["raw_memb"], data["seg_memb"]  #
        else:
            x, nuc = data[0:2]  #
        #  go through the network
        start_time = time.time()
        pred_bin = model(x)
        elapsed_time = time.time() - start_time
        runtimes.append(elapsed_time)
        #  Regression only has one channel
        if pred_bin.shape[1] > 1:
            pred_bin = delete_isolate_labels(pred_bin)
            pred_bin = pred_bin.argmax(1) # [channel, height, width, depth]

        #  binary prediction
        pred_bin = pred_bin.cpu().numpy()
        pred_bin = pred_bin.squeeze().transpose([1, 2, 0]) # from z, x, y to x, y ,z
        pred_bin = resize(pred_bin.astype(float), size, mode='constant', cval=0, order=0, anti_aliasing=False)

        #  post process
        if postprocess == True:
            pass  #

        #  save volume and snapshot data
        if snapsot is not None:
            x =  x.cpu().numpy()
            x = x.transpose([0, 1, 3, 4, 2])
            image_dict = dict(Raw=x[0, 0, :, 100, :], output_bin=pred_bin[:, 100, :])
            snapsot.show_current_images(image_dict)
        if savepath is not None:
            if "npy" in save_format.lower():
                np.save(os.path.join(savepath,  names[i].split("_")[0], "SegMemb", names[i] + "_segMemb"), pred_bin)
            elif "nii.gz" in save_format.lower():
                save_name = os.path.join(savepath, names[i].split("_")[0],  "SegMemb",  names[i] + "_segMemb.nii.gz")
                # pred_bin_saving=np.zeros(pred_bin.shape)
                # pred_bin_saving[pred_bin>(14/15)]=1
                # nib_save(pred_bin,save_name)

                nib_save((pred_bin*256).astype(np.int16), save_name) # pred_bin is range(0,1)

def membrane2cell(args):
    for embryo_name in args.running_embryos:
        # get the binary mask of a embryo with histogram transform(3DMMNS)
        # embryo_mask = get_eggshell(embryo_name,root_folder=args.running_data_dir)
        embryo_mask=None

        file_names = glob.glob(os.path.join(args.running_data_dir,embryo_name, "SegMemb",'*.nii.gz'))
        parameters = []
        for file_name in file_names:
            parameters.append([embryo_name, file_name, embryo_mask,args.running_data_dir])
            # segment_membrane([embryo_name, file_name, embryo_mask])
        mpPool = mp.Pool(8)
        print('started ', 8, ' processes ',mpPool)
        for _ in tqdm(mpPool.imap_unordered(segment_membrane, parameters), total=len(parameters), desc="{} edt cell membrane --> single cell instance".format(embryo_name)):
            pass

def combine_dividing_cells(args):
    combine_division(args.running_embryos, args.run_max_times, args.running_data_dir,overwrite=False)



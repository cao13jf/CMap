#  import dependency library
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

# import user defined
from .data_utils import get_all_stack, pkload
from .transforms import Compose, RandCrop, RandomFlip, NumpyType, RandomRotation, Pad, Resize, ContourEDT, RandomIntensityChange
from .augmentations import contour_distance

#=======================================
#  Import membrane datasets
#=======================================
#   data format: dict([raw_memb, raw_nuc, seg_nuc, 'seg_memb, seg_cell'])
class Memb3DDataset(Dataset):
    def __init__(self, root="dataset/train", membrane_names=None, for_train=True, return_target=True, transforms=None, suffix="*.pkl"):
        if membrane_names is None:
            membrane_names = [name for name in os.listdir(root) if os.path.isdir(os.path.join(root, name))]
        self.paths = get_all_stack(root, membrane_names, suffix=suffix)
        self.names = [os.path.basename(path).split(".")[0] for path in self.paths]
        self.for_train = for_train
        self.return_target = return_target
        self.transforms = eval(transforms or "Identity()")  # TODO: define transformation library

    def __getitem__(self, item):
        stack_name = self.names[item]
        tp = float(stack_name.split("_")[1][1:])
        load_dict = pkload(self.paths[item])  # Choose whether to need nucleus stack
        if self.return_target:
            seg_nuc = load_dict["seg_nuc"]
            edt_nuc = contour_distance(seg_nuc, d_threshold=60)
            raw, seg, edt_nuc = self.transforms([load_dict["raw_memb"], load_dict["seg_memb"], edt_nuc])
            seg = seg[np.newaxis, np.newaxis, ...].transpose([0, 1, 4, 2, 3])  #[Batchsize, Depth, Height, Width]
            seg = np.ascontiguousarray(seg)
            edt_nuc = edt_nuc[np.newaxis, np.newaxis, ...].transpose([0, 1, 4, 2, 3])  #[Batchsize, Depth, Height, Width]
            edt_nuc = np.ascontiguousarray(edt_nuc)
        else:
            raw = self.transforms(load_dict["raw_memb"])
        raw = raw[np.newaxis, np.newaxis, :, :, :]  # [Batchsize, channels, Height, Width, Depth]
        raw = np.ascontiguousarray(raw.transpose([0, 1, 4, 2, 3]))  # [Batchsize, channels, Depth, Height, Width]
        raw = torch.from_numpy(raw)

        #==================================== add time information =======================
        tp = tp * torch.ones_like(raw) / 200.0
        raw = torch.cat([raw, tp, torch.from_numpy(edt_nuc)], dim=1)
        #==================================== add time information =======================
        #==================================== add nucleus distance channel================
        if self.return_target:
            seg = torch.from_numpy(seg)
            return raw, seg
        else:
            return raw

    def __len__(self):
        return len(self.names)

    def collate(self, batch):
        out_batch =  [torch.cat(v) for v in zip(*batch)]
        if len(batch) == 1:
            out_batch = [x.unsqueeze(0) for x in out_batch]
        return out_batch

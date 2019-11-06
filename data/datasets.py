#  import dependency library
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

# import user defined
from .data_utils import get_all_stack, pkload
from .transforms import Compose, RandCrop, RandomFlip, NumpyType, RandomRotation, Pad, Resize, ContourEDT, RandomIntensityChange
from .augmentations import contour_distance, contour_distance_outside_negative

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
        seg_nuc = load_dict["seg_nuc"]
        edt_nuc = contour_distance(seg_nuc, d_threshold=2)
        if self.return_target:
            target_distance = contour_distance_outside_negative(load_dict["seg_memb"], load_dict["seg_cell"], d_threshold=15)
            raw, seg_dis, seg_bin, edt_nuc = self.transforms([load_dict["raw_memb"], target_distance, load_dict["seg_memb"], edt_nuc])
            raw, seg_dis, seg_bin, edt_nuc = self.volume2tensor([raw, seg_dis, seg_bin, edt_nuc])
        else:
            raw, edt_nuc = self.transforms([load_dict["raw_memb"], edt_nuc])
            raw, edt_nuc = self.volume2tensor([raw, edt_nuc])

        #==================================== add time information =======================
        # tp = tp * torch.ones_like(raw) / 200.0
        raw = torch.cat([raw, edt_nuc], dim=1)
        #==================================== add time information =======================
        #==================================== add nucleus distance channel================
        if self.return_target:
            return raw, seg_dis, seg_bin
        else:
            return raw
    def volume2tensor(self, volumes, dim_order = [2, 0, 1]):
        volumes = volumes if isinstance(volumes, list) else [volumes]
        outputs = []
        for volume in volumes:
            volume = volume.transpose(dim_order)[np.newaxis, np.newaxis, ...]
            volume = np.ascontiguousarray(volume)
            volume = torch.from_numpy(volume)
            outputs.append(volume)

        return outputs

    def __len__(self):
        return len(self.names)

    def collate(self, batch):
        out_batch =  [torch.cat(v) for v in zip(*batch)]
        if len(batch) == 1:
            out_batch = [x.unsqueeze(0) for x in out_batch]
        return out_batch

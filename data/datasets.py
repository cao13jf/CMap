#  import dependency library
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

# import user defined
from .data_utils import get_all_stack, pkload
from .transforms import Compose, RandCrop, RandomFlip, NumpyType, RandomRotation, Pad, Resize


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
        load_dict = pkload(self.paths[item])  # Choose whether to need nucleus stack
        if self.return_target:
            raw, seg = self.transforms([load_dict["raw_memb"], load_dict["seg_memb"]])
            seg = seg[np.newaxis, ...].transpose([0, 3, 1, 2])  #[Batchsize, Depth, Height, Width]
            seg = np.ascontiguousarray(seg)
        else:
            raw = self.transforms(load_dict["raw_memb"])
        raw = raw[np.newaxis, np.newaxis, :, :, :]  # [Batchsize, channels, Height, Width, Depth]
        raw = np.ascontiguousarray(raw.transpose([0, 1, 4, 2, 3]))  # [Batchsize, channels, Depth, Height, Width]

        if self.return_target:
            raw, seg = torch.from_numpy(raw), torch.from_numpy(seg)
            return raw, seg
        else:
            raw = torch.from_numpy(raw)
            return raw

    def __len__(self):
        return len(self.names)

    def collate(self, batch):
        out_batch =  [torch.cat(v) for v in zip(*batch)]
        if len(batch) == 1:
            out_batch = [x.unsqueeze(0) for x in out_batch]
        return out_batch

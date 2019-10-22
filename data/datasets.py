#  import dependency library
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

# import user defined
from .data_utils import get_all_stack, pkload
from .augmentations import cell_sliced_distance, regression_to_class, sampled_cell_mask
from .transforms import Compose, RandCrop, RandomFlip, NumpyType, RandomRotation, Pad, Resize


#=======================================
#  Import membrane datasets
#=======================================
#   data format: dict([raw_memb, raw_nuc, seg_nuc, 'seg_memb, seg_cell'])
#   "weak dataset" where unanotated datasets are masked out
class Memb3DDataset(Dataset):
    def __init__(self, root="dataset/train", membrane_names=None, out_class=10, for_train=True, return_target=True, transforms=None, suffix="*.pkl", args=None):
        if membrane_names is None:
            membrane_names = [name for name in os.listdir(root) if os.path.isdir(os.path.join(root, name))]
        self.paths = get_all_stack(root, membrane_names, suffix=suffix)
        self.names = [os.path.basename(path).split(".")[0] for path in self.paths]
        self.for_train = for_train
        self.return_target = return_target
        if args is not None:
            self.out_class = out_class
            self.d_threshold = args.d_threshold
            self.d_uniform = args.d_uniform
        self.transforms = eval(transforms or "Identity()")  # TODO: define transformation library

    def __getitem__(self, item):
        stack_name = self.names[item]

        load_dict = pkload(self.paths[item])  # Choose whether to need nucleus stack
        if self.return_target:
            seg, mask = cell_sliced_distance(load_dict["seg_cell"], load_dict["seg_memb"], sampled=self.d_uniform, d_threshold=self.d_threshold)
            seg = regression_to_class(seg, self.out_class, uniform=self.d_uniform)
            # seg, mask = sampled_cell_mask(load_dict["seg_cell"], load_dict["seg_memb"])
            raw, seg, mask = self.transforms([load_dict["raw_memb"], seg, mask])
            seg = seg[np.newaxis, ...].transpose([0, 3, 1, 2])  #[Batchsize, Depth, Height, Width]
            mask = mask[np.newaxis, ...].transpose([0, 3, 1, 2])  #[Batchsize, Depth, Height, Width]
            seg = np.ascontiguousarray(seg)
            mask = np.ascontiguousarray(mask)
        else:
            raw = self.transforms(load_dict["raw_memb"])
        raw = raw[np.newaxis, np.newaxis, :, :, :]  # [Batchsize, channels, Height, Width, Depth]
        # time_channel = np.ones_like(raw) * float(stack_name.split("_")[1][1:])
        # raw = np.concatenate((raw, time_channel), axis=1)
        raw = np.ascontiguousarray(raw.transpose([0, 1, 4, 2, 3]))  # [Batchsize, channels, Depth, Height, Width]

        if self.return_target:
            raw, seg, mask = torch.from_numpy(raw), torch.from_numpy(seg), torch.from_numpy(mask)
            return raw, seg, mask
        else:
            raw = torch.from_numpy(raw)
            return raw

    def __len__(self):
        return len(self.names)

    def collate(self, batch):
        out_batch =  [torch.cat(v) for v in zip(*batch)]
        # if len(batch) == 1:
        #     out_batch = [x.unsqueeze(0) for x in out_batch]
        return out_batch

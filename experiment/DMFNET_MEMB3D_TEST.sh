#!/usr/bin/env bash

set -x

# python preprocess.py

python test.py --cfg DMFNET_MEMB3D_TRAIN --trained_model model_last.pth --mode 2 --gpu 0 --suffix *.pkl --save_format nii.gz
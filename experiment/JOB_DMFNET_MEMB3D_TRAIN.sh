#!/bin/bash
#!/usr/bin/env bash

#SBATCH -J Memb_3D_WEAKDIS
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --gres=gpu:4
#SBATCH --nodelist=node3

set -ex
srun bash -c "python test.py --gpu=0,1,2,3 --cfg DMFNET_MEMB3D_TRAIN --batch_size 8 --show_image_freq -1 --show_loss_freq -1"
wait

set -x

python train.py --cfg DMFNET_MEMB3D_TRAIN --gpu 0 --batch_size 1 --restore '' --show_image_freq 10 --show_loss_freq 5
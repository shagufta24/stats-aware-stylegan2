#!/bin/bash
source /etc/profile
module load cuda-toolkit/11.1
#export CUDA_HOME=/software/cuda-11.1/
module load gcc/9.2.0
gcc --version
source ~/.bashrc
conda activate debug_tst_env
cd ~/stats_aware_gans/stats-aware-stylegan2-ada
NCCL_P2P_LEVEL=NVL python train.py --outdir=/home/nkamath5/stats_aware_gans/training_runs_stats_aware_gans --data=/shared/rsaas/nkamath5/train_128.zip --aug=noaug --cfg auto --gamma=2 --gpus=1 --batch=128 --snap=75 \
--tst_pl_wts_ratio 0.05  --features all --kimg 10000 --num_random_runs 20000 -loss Bhatt \
--resume=/home/nkamath5/stats_aware_gans/training_runs_stats_aware_gans/00056-train_128-auto1-gamma2-kimg15000-batch128-noaug/network-snapshot-006758.pkl
#--alphas 0.02
#--resume=/home/nkamath5/stats_aware_gans/training_runs_stats_aware_gans/00037-train_128-auto1-gamma2-kimg15000-batch128-noaug/network-snapshot-005529.pkl
# seed=28

#NCCL_P2P_LEVEL=NVL python train.py --outdir=/home/nkamath5/stats_aware_gans/debug_training_runs_stats_aware_gans --data=/shared/rsaas/nkamath5/train_128.zip --aug=noaug --cfg auto --gamma=2 --gpus=1 --batch=320 --snap=75 --tst_pl_wts_ratio 1 --alphas 0.01 --features all --kimg 10000 --no_lazy_reg_g --grad_accu_rounds 5

##NCCL_P2P_LEVEL=NVL python train.py --outdir=/home/nkamath5/stats_aware_gans/training_runs_stats_aware_gans --data=/shared/rsaas/nkamath5/challenge_data.zip --aug=noaug --cfg paper512 --gamma=100 --gpus=1 --batch=16 --snap=75 \
##--alphas 0.00001 --tst_pl_wts_ratio 1 --features all

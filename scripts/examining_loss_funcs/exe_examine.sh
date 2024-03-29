#!/bin/bash
source /etc/profile
source ~/.bashrc
conda activate debug_tst_env
cd /home/nkamath5/stats_aware_gans/stats-aware-stylegan2-ada/scripts/examining_loss_funcs
python determining_right_setup_for_loss_func.py KL --num_cosines 200

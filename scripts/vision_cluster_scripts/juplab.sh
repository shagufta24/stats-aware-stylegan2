#!/bin/bash
cd ~
module load gcc/9.2.0
export CUDA_HOME=/software/cuda-11.1/
source ~/.bashrc
conda activate eval_env
jupyter-lab --no-browser --ip=0.0.0.0


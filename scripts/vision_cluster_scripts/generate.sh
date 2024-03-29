#!/bin/bash
source /etc/profile
module load cuda-toolkit/11.1
#module load cuda-toolkit/11.0
#export CUDA_HOME=/software/cuda-11.1/
module load gcc/9.2.0
gcc --version
source ~/.bashrc
conda activate tst_env
cd /home/nkamath5/stats_aware_gans/stats-aware-stylegan2-ada
#python generate.py --outdir=/home/nkamath5/stats_aware_gans/aapm_generated_images/00027-128-auto1-gamma2-batch64-kimg000604 --seeds=0-10000 --network=/home/nkamath5/stats_aware_gans/debug_training_runs_stats_aware_gans/00024-train_128-auto1-gamma2-kimg10000-batch64-noaug/network-snapshot-000604.pkl
#python generate.py --outdir=/home/nkamath5/stats_aware_gans/aapm_generated_images/00026-128-auto1-gamma2-batch64-kimg000604 --seeds=0-10000 --network=/home/nkamath5/stats_aware_gans/training_runs_stats_aware_gans/00026-train_128-auto1-gamma2-batch64-noaug-resumecustom/network-snapshot-000604.pkl

#python generate.py --outdir=/home/nkamath5/stats_aware_gans/aapm_generated_images/new/00029-128-auto1-gamma2-batch64-kimg000000 --seeds=0-10000 --network=/home/nkamath5/stats_aware_gans/training_runs_stats_aware_gans/00029-train_128-auto1-gamma2-batch64-noaug/network-snapshot-000000.pkl
#python generate.py --outdir=/home/nkamath5/stats_aware_gans/aapm_generated_images/new/00029-128-auto1-gamma2-batch64-kimg000907 --seeds=0-10000 --network=/home/nkamath5/stats_aware_gans/training_runs_stats_aware_gans/00029-train_128-auto1-gamma2-batch64-noaug/network-snapshot-000907.pkl
#python generate.py --outdir=/home/nkamath5/stats_aware_gans/aapm_generated_images/new/00029-128-auto1-gamma2-batch64-kimg001814 --seeds=0-10000 --network=/home/nkamath5/stats_aware_gans/training_runs_stats_aware_gans/00029-train_128-auto1-gamma2-batch64-noaug/network-snapshot-001814.pkl
#python generate.py --outdir=/home/nkamath5/stats_aware_gans/aapm_generated_images/new/00029-128-auto1-gamma2-batch64-kimg002721 --seeds=0-10000 --network=/home/nkamath5/stats_aware_gans/training_runs_stats_aware_gans/00029-train_128-auto1-gamma2-batch64-noaug/network-snapshot-002721.pkl
#python generate.py --outdir=/home/nkamath5/stats_aware_gans/aapm_generated_images/new/00029-128-auto1-gamma2-batch64-kimg003628 --seeds=0-10000 --network=/home/nkamath5/stats_aware_gans/training_runs_stats_aware_gans/00029-train_128-auto1-gamma2-batch64-noaug/network-snapshot-003628.pkl
#python generate.py --outdir=/home/nkamath5/stats_aware_gans/aapm_generated_images/new/00029-128-auto1-gamma2-batch64-kimg004536 --seeds=0-10000 --network=/home/nkamath5/stats_aware_gans/training_runs_stats_aware_gans/00029-train_128-auto1-gamma2-batch64-noaug/network-snapshot-004536.pkl
#python generate.py --outdir=/home/nkamath5/stats_aware_gans/aapm_generated_images/new/00029-128-auto1-gamma2-batch64-kimg005443 --seeds=0-10000 --network=/home/nkamath5/stats_aware_gans/training_runs_stats_aware_gans/00029-train_128-auto1-gamma2-batch64-noaug/network-snapshot-005443.pkl
#python generate.py --outdir=/home/nkamath5/stats_aware_gans/aapm_generated_images/new/00029-128-auto1-gamma2-batch64-kimg006350 --seeds=0-10000 --network=/home/nkamath5/stats_aware_gans/training_runs_stats_aware_gans/00029-train_128-auto1-gamma2-batch64-noaug/network-snapshot-006350.pkl
#python generate.py --outdir=/home/nkamath5/stats_aware_gans/aapm_generated_images/new/00029-128-auto1-gamma2-batch64-kimg007257 --seeds=0-10000 --network=/home/nkamath5/stats_aware_gans/training_runs_stats_aware_gans/00029-train_128-auto1-gamma2-batch64-noaug/network-snapshot-007257.pkl
#python generate.py --outdir=/home/nkamath5/stats_aware_gans/aapm_generated_images/new/00029-128-auto1-gamma2-batch64-kimg008164 --seeds=0-10000 --network=/home/nkamath5/stats_aware_gans/training_runs_stats_aware_gans/00029-train_128-auto1-gamma2-batch64-noaug/network-snapshot-008164.pkl
#python generate.py --outdir=/home/nkamath5/stats_aware_gans/aapm_generated_images/new/00029-128-auto1-gamma2-batch64-kimg009072 --seeds=0-10000 --network=/home/nkamath5/stats_aware_gans/training_runs_stats_aware_gans/00029-train_128-auto1-gamma2-batch64-noaug/network-snapshot-009072.pkl
#python generate.py --outdir=/home/nkamath5/stats_aware_gans/aapm_generated_images/new/00029-128-auto1-gamma2-batch64-kimg009979 --seeds=0-10000 --network=/home/nkamath5/stats_aware_gans/training_runs_stats_aware_gans/00029-train_128-auto1-gamma2-batch64-noaug/network-snapshot-009979.pkl
#python generate.py --outdir=/home/nkamath5/stats_aware_gans/aapm_generated_images/new/00029-128-auto1-gamma2-batch64-kimg010886 --seeds=0-10000 --network=/home/nkamath5/stats_aware_gans/training_runs_stats_aware_gans/00029-train_128-auto1-gamma2-batch64-noaug/network-snapshot-010886.pkl
#python generate.py --outdir=/home/nkamath5/stats_aware_gans/aapm_generated_images/new/00029-128-auto1-gamma2-batch64-kimg011793 --seeds=0-10000 --network=/home/nkamath5/stats_aware_gans/training_runs_stats_aware_gans/00029-train_128-auto1-gamma2-batch64-noaug/network-snapshot-011793.pkl
#python generate.py --outdir=/home/nkamath5/stats_aware_gans/aapm_generated_images/new/00029-128-auto1-gamma2-batch64-kimg012700 --seeds=0-10000 --network=/home/nkamath5/stats_aware_gans/training_runs_stats_aware_gans/00029-train_128-auto1-gamma2-batch64-noaug/network-snapshot-012700.pkl
#python generate.py --outdir=/home/nkamath5/stats_aware_gans/aapm_generated_images/new/00029-128-auto1-gamma2-batch64-kimg013608 --seeds=0-10000 --network=/home/nkamath5/stats_aware_gans/training_runs_stats_aware_gans/00029-train_128-auto1-gamma2-batch64-noaug/network-snapshot-013608.pkl
#python generate.py --outdir=/home/nkamath5/stats_aware_gans/aapm_generated_images/new/00029-128-auto1-gamma2-batch64-kimg014515 --seeds=0-10000 --network=/home/nkamath5/stats_aware_gans/training_runs_stats_aware_gans/00029-train_128-auto1-gamma2-batch64-noaug/network-snapshot-014515.pkl

#python generate.py --outdir=/data/nkamath5/stats_aware_gans/aapm_generated_images/00029-128-auto1-gamma2-batch64-kimg000302-125K --seeds=0-125000 --network=/home/nkamath5/stats_aware_gans/training_runs_stats_aware_gans/00029-train_128-auto1-gamma2-batch64-noaug/network-snapshot-000302.pkl
python generate.py --outdir=/data/nkamath5/stats_aware_gans/aapm_generated_images/00029-128-auto1-gamma2-batch64-kimg002721-125K --seeds=0-125000 --network=/home/nkamath5/stats_aware_gans/training_runs_stats_aware_gans/00029-train_128-auto1-gamma2-batch64-noaug/network-snapshot-002721.pkl
#python generate.py --outdir=/data/nkamath5/stats_aware_gans/aapm_generated_images/00029-128-auto1-gamma2-batch64-kimg008467-125K --seeds=0-125000 --network=/home/nkamath5/stats_aware_gans/training_runs_stats_aware_gans/00029-train_128-auto1-gamma2-batch64-noaug/network-snapshot-008467.pkl
#python generate.py --outdir=/data/nkamath5/stats_aware_gans/aapm_generated_images/00029-128-auto1-gamma2-batch64-kimg014212-125K --seeds=0-125000 --network=/home/nkamath5/stats_aware_gans/training_runs_stats_aware_gans/00029-train_128-auto1-gamma2-batch64-noaug/network-snapshot-014212.pkl

#python generate.py --outdir=/data/nkamath5/stats_aware_gans/aapm_generated_images/00030-128-auto1-gamma2-batch64-kimg000302-125K --seeds=0-125000 --network=/home/nkamath5/stats_aware_gans/training_runs_stats_aware_gans/00030-train_128-auto1-gamma2-kimg10000-batch64-noaug/network-snapshot-000302.pkl
#python generate.py --outdir=/data/nkamath5/stats_aware_gans/aapm_generated_images/00030-128-auto1-gamma2-batch64-kimg002721-125K --seeds=0-125000 --network=/home/nkamath5/stats_aware_gans/training_runs_stats_aware_gans/00030-train_128-auto1-gamma2-kimg10000-batch64-noaug/network-snapshot-002721.pkl
#python generate.py --outdir=/data/nkamath5/stats_aware_gans/aapm_generated_images/00030-128-auto1-gamma2-batch64-kimg008467-125K --seeds=0-125000 --network=/home/nkamath5/stats_aware_gans/training_runs_stats_aware_gans/00030-train_128-auto1-gamma2-kimg10000-batch64-noaug/network-snapshot-008467.pkl

#python generate.py --outdir=/home/nkamath5/stats_aware_gans/aapm_generated_images/new/00030-128-auto1-gamma2-batch64-kimg000907 --seeds=0-10000 --network=/home/nkamath5/stats_aware_gans/training_runs_stats_aware_gans/00030-train_128-auto1-gamma2-kimg10000-batch64-noaug/network-snapshot-000907.pkl
#python generate.py --outdir=/home/nkamath5/stats_aware_gans/aapm_generated_images/new/00030-128-auto1-gamma2-batch64-kimg001814 --seeds=0-10000 --network=/home/nkamath5/stats_aware_gans/training_runs_stats_aware_gans/00030-train_128-auto1-gamma2-kimg10000-batch64-noaug/network-snapshot-001814.pkl
#python generate.py --outdir=/home/nkamath5/stats_aware_gans/aapm_generated_images/new/00030-128-auto1-gamma2-batch64-kimg002721 --seeds=0-10000 --network=/home/nkamath5/stats_aware_gans/training_runs_stats_aware_gans/00030-train_128-auto1-gamma2-kimg10000-batch64-noaug/network-snapshot-002721.pkl
#python generate.py --outdir=/home/nkamath5/stats_aware_gans/aapm_generated_images/new/00030-128-auto1-gamma2-batch64-kimg003628 --seeds=0-10000 --network=/home/nkamath5/stats_aware_gans/training_runs_stats_aware_gans/00030-train_128-auto1-gamma2-kimg10000-batch64-noaug/network-snapshot-003628.pkl
#python generate.py --outdir=/home/nkamath5/stats_aware_gans/aapm_generated_images/new/00030-128-auto1-gamma2-batch64-kimg004536 --seeds=0-10000 --network=/home/nkamath5/stats_aware_gans/training_runs_stats_aware_gans/00030-train_128-auto1-gamma2-kimg10000-batch64-noaug/network-snapshot-004536.pkl
#python generate.py --outdir=/home/nkamath5/stats_aware_gans/aapm_generated_images/new/00030-128-auto1-gamma2-batch64-kimg005443 --seeds=0-10000 --network=/home/nkamath5/stats_aware_gans/training_runs_stats_aware_gans/00030-train_128-auto1-gamma2-kimg10000-batch64-noaug/network-snapshot-005443.pkl
#python generate.py --outdir=/home/nkamath5/stats_aware_gans/aapm_generated_images/new/00030-128-auto1-gamma2-batch64-kimg006350 --seeds=0-10000 --network=/home/nkamath5/stats_aware_gans/training_runs_stats_aware_gans/00030-train_128-auto1-gamma2-kimg10000-batch64-noaug/network-snapshot-006350.pkl
#python generate.py --outdir=/home/nkamath5/stats_aware_gans/aapm_generated_images/new/00030-128-auto1-gamma2-batch64-kimg007257 --seeds=0-10000 --network=/home/nkamath5/stats_aware_gans/training_runs_stats_aware_gans/00030-train_128-auto1-gamma2-kimg10000-batch64-noaug/network-snapshot-007257.pkl
#python generate.py --outdir=/home/nkamath5/stats_aware_gans/aapm_generated_images/new/00030-128-auto1-gamma2-batch64-kimg008164 --seeds=0-10000 --network=/home/nkamath5/stats_aware_gans/training_runs_stats_aware_gans/00030-train_128-auto1-gamma2-kimg10000-batch64-noaug/network-snapshot-008164.pkl
#python generate.py --outdir=/home/nkamath5/stats_aware_gans/aapm_generated_images/new/00030-128-auto1-gamma2-batch64-kimg009072 --seeds=0-10000 --network=/home/nkamath5/stats_aware_gans/training_runs_stats_aware_gans/00030-train_128-auto1-gamma2-kimg10000-batch64-noaug/network-snapshot-009072.pkl
#python generate.py --outdir=/home/nkamath5/stats_aware_gans/aapm_generated_images/new/00030-128-auto1-gamma2-batch64-kimg009979 --seeds=0-10000 --network=/home/nkamath5/stats_aware_gans/training_runs_stats_aware_gans/00030-train_128-auto1-gamma2-kimg10000-batch64-noaug/network-snapshot-009979.pkl

#python generate.py --outdir=/home/nkamath5/stats_aware_gans/aapm_generated_images/new/00031-128-auto1-gamma2-batch64-kimg000907 --seeds=0-10000 --network=/home/nkamath5/stats_aware_gans/training_runs_stats_aware_gans/00031-train_128-auto1-gamma2-kimg10000-batch64-noaug/network-snapshot-000907.pkl
#python generate.py --outdir=/home/nkamath5/stats_aware_gans/aapm_generated_images/new/00031-128-auto1-gamma2-batch64-kimg001814 --seeds=0-10000 --network=/home/nkamath5/stats_aware_gans/training_runs_stats_aware_gans/00031-train_128-auto1-gamma2-kimg10000-batch64-noaug/network-snapshot-001814.pkl
#python generate.py --outdir=/home/nkamath5/stats_aware_gans/aapm_generated_images/new/00031-128-auto1-gamma2-batch64-kimg002721 --seeds=0-10000 --network=/home/nkamath5/stats_aware_gans/training_runs_stats_aware_gans/00031-train_128-auto1-gamma2-kimg10000-batch64-noaug/network-snapshot-002721.pkl
#python generate.py --outdir=/home/nkamath5/stats_aware_gans/aapm_generated_images/new/00031-128-auto1-gamma2-batch64-kimg003628 --seeds=0-10000 --network=/home/nkamath5/stats_aware_gans/training_runs_stats_aware_gans/00031-train_128-auto1-gamma2-kimg10000-batch64-noaug/network-snapshot-003628.pkl
#python generate.py --outdir=/home/nkamath5/stats_aware_gans/aapm_generated_images/new/00031-128-auto1-gamma2-batch64-kimg004536 --seeds=0-10000 --network=/home/nkamath5/stats_aware_gans/training_runs_stats_aware_gans/00031-train_128-auto1-gamma2-kimg10000-batch64-noaug/network-snapshot-004536.pkl
#python generate.py --outdir=/home/nkamath5/stats_aware_gans/aapm_generated_images/new/00031-128-auto1-gamma2-batch64-kimg005443 --seeds=0-10000 --network=/home/nkamath5/stats_aware_gans/training_runs_stats_aware_gans/00031-train_128-auto1-gamma2-kimg10000-batch64-noaug/network-snapshot-005443.pkl
#python generate.py --outdir=/home/nkamath5/stats_aware_gans/aapm_generated_images/new/00031-128-auto1-gamma2-batch64-kimg006350 --seeds=0-10000 --network=/home/nkamath5/stats_aware_gans/training_runs_stats_aware_gans/00031-train_128-auto1-gamma2-kimg10000-batch64-noaug/network-snapshot-006350.pkl
#python generate.py --outdir=/home/nkamath5/stats_aware_gans/aapm_generated_images/new/00031-128-auto1-gamma2-batch64-kimg007257 --seeds=0-10000 --network=/home/nkamath5/stats_aware_gans/training_runs_stats_aware_gans/00031-train_128-auto1-gamma2-kimg10000-batch64-noaug/network-snapshot-007257.pkl
#python generate.py --outdir=/home/nkamath5/stats_aware_gans/aapm_generated_images/new/00031-128-auto1-gamma2-batch64-kimg008164 --seeds=0-10000 --network=/home/nkamath5/stats_aware_gans/training_runs_stats_aware_gans/00031-train_128-auto1-gamma2-kimg10000-batch64-noaug/network-snapshot-008164.pkl
#python generate.py --outdir=/home/nkamath5/stats_aware_gans/aapm_generated_images/new/00031-128-auto1-gamma2-batch64-kimg009072 --seeds=0-10000 --network=/home/nkamath5/stats_aware_gans/training_runs_stats_aware_gans/00031-train_128-auto1-gamma2-kimg10000-batch64-noaug/network-snapshot-009072.pkl
#python generate.py --outdir=/home/nkamath5/stats_aware_gans/aapm_generated_images/new/00031-128-auto1-gamma2-batch64-kimg009979 --seeds=0-10000 --network=/home/nkamath5/stats_aware_gans/training_runs_stats_aware_gans/00031-train_128-auto1-gamma2-kimg10000-batch64-noaug/network-snapshot-009979.pkl

#python generate.py --outdir=/home/nkamath5/stats_aware_gans/aapm_generated_images/new/00032-128-auto1-gamma2-batch64-abskimg000907 --seeds=0-10000 --network=/home/nkamath5/stats_aware_gans/training_runs_stats_aware_gans/00032-train_128-auto1-gamma2-kimg10000-batch64-noaug-resumecustom/network-snapshot-000907.pkl
#python generate.py --outdir=/home/nkamath5/stats_aware_gans/aapm_generated_images/new/00032-128-auto1-gamma2-batch64-abskimg001814 --seeds=0-10000 --network=/home/nkamath5/stats_aware_gans/training_runs_stats_aware_gans/00032-train_128-auto1-gamma2-kimg10000-batch64-noaug-resumecustom/network-snapshot-001814.pkl
#python generate.py --outdir=/home/nkamath5/stats_aware_gans/aapm_generated_images/new/00032-128-auto1-gamma2-batch64-abskimg002721 --seeds=0-10000 --network=/home/nkamath5/stats_aware_gans/training_runs_stats_aware_gans/00032-train_128-auto1-gamma2-kimg10000-batch64-noaug-resumecustom/network-snapshot-002721.pkl
#python generate.py --outdir=/home/nkamath5/stats_aware_gans/aapm_generated_images/new/00032-128-auto1-gamma2-batch64-abskimg003628 --seeds=0-10000 --network=/home/nkamath5/stats_aware_gans/training_runs_stats_aware_gans/00032-train_128-auto1-gamma2-kimg10000-batch64-noaug-resumecustom/network-snapshot-003628.pkl
#python generate.py --outdir=/home/nkamath5/stats_aware_gans/aapm_generated_images/new/00032-128-auto1-gamma2-batch64-abskimg004536 --seeds=0-10000 --network=/home/nkamath5/stats_aware_gans/training_runs_stats_aware_gans/00032-train_128-auto1-gamma2-kimg10000-batch64-noaug-resumecustom/network-snapshot-004536.pkl
#python generate.py --outdir=/home/nkamath5/stats_aware_gans/aapm_generated_images/new/00032-128-auto1-gamma2-batch64-abskimg005443 --seeds=0-10000 --network=/home/nkamath5/stats_aware_gans/training_runs_stats_aware_gans/00032-train_128-auto1-gamma2-kimg10000-batch64-noaug-resumecustom/network-snapshot-005443.pkl

#python generate.py --outdir=/home/nkamath5/stats_aware_gans/aapm_generated_images/new/00034-128-auto1-gamma2-batch64-abskimg009071 --seeds=0-10000 --network=/home/nkamath5/stats_aware_gans/training_runs_stats_aware_gans/00034-train_128-auto1-gamma2-kimg10000-batch64-noaug-resumecustom/network-snapshot-000907.pkl
#python generate.py --outdir=/home/nkamath5/stats_aware_gans/aapm_generated_images/new/00034-128-auto1-gamma2-batch64-abskimg009978 --seeds=0-10000 --network=/home/nkamath5/stats_aware_gans/training_runs_stats_aware_gans/00034-train_128-auto1-gamma2-kimg10000-batch64-noaug-resumecustom/network-snapshot-001814.pkl
#python generate.py --outdir=/home/nkamath5/stats_aware_gans/aapm_generated_images/new/00034-128-auto1-gamma2-batch64-abskimg010885 --seeds=0-10000 --network=/home/nkamath5/stats_aware_gans/training_runs_stats_aware_gans/00034-train_128-auto1-gamma2-kimg10000-batch64-noaug-resumecustom/network-snapshot-002721.pkl
#python generate.py --outdir=/home/nkamath5/stats_aware_gans/aapm_generated_images/new/00034-128-auto1-gamma2-batch64-abskimg011792 --seeds=0-10000 --network=/home/nkamath5/stats_aware_gans/training_runs_stats_aware_gans/00034-train_128-auto1-gamma2-kimg10000-batch64-noaug-resumecustom/network-snapshot-003628.pkl
#python generate.py --outdir=/home/nkamath5/stats_aware_gans/aapm_generated_images/new/00034-128-auto1-gamma2-batch64-abskimg012699 --seeds=0-10000 --network=/home/nkamath5/stats_aware_gans/training_runs_stats_aware_gans/00034-train_128-auto1-gamma2-kimg10000-batch64-noaug-resumecustom/network-snapshot-004536.pkl
#python generate.py --outdir=/home/nkamath5/stats_aware_gans/aapm_generated_images/new/00034-128-auto1-gamma2-batch64-abskimg013606 --seeds=0-10000 --network=/home/nkamath5/stats_aware_gans/training_runs_stats_aware_gans/00034-train_128-auto1-gamma2-kimg10000-batch64-noaug-resumecustom/network-snapshot-005443.pkl
#python generate.py --outdir=/home/nkamath5/stats_aware_gans/aapm_generated_images/new/00034-128-auto1-gamma2-batch64-abskimg014513 --seeds=0-10000 --network=/home/nkamath5/stats_aware_gans/training_runs_stats_aware_gans/00034-train_128-auto1-gamma2-kimg10000-batch64-noaug-resumecustom/network-snapshot-006350.pkl
#python generate.py --outdir=/home/nkamath5/stats_aware_gans/aapm_generated_images/new/00034-128-auto1-gamma2-batch64-abskimg015420 --seeds=0-10000 --network=/home/nkamath5/stats_aware_gans/training_runs_stats_aware_gans/00034-train_128-auto1-gamma2-kimg10000-batch64-noaug-resumecustom/network-snapshot-007257.pkl
#python generate.py --outdir=/home/nkamath5/stats_aware_gans/aapm_generated_images/new/00034-128-auto1-gamma2-batch64-abskimg016327 --seeds=0-10000 --network=/home/nkamath5/stats_aware_gans/training_runs_stats_aware_gans/00034-train_128-auto1-gamma2-kimg10000-batch64-noaug-resumecustom/network-snapshot-008164.pkl
#python generate.py --outdir=/home/nkamath5/stats_aware_gans/aapm_generated_images/new/00034-128-auto1-gamma2-batch64-abskimg017234 --seeds=0-10000 --network=/home/nkamath5/stats_aware_gans/training_runs_stats_aware_gans/00034-train_128-auto1-gamma2-kimg10000-batch64-noaug-resumecustom/network-snapshot-009072.pkl

#python generate.py --outdir=/home/nkamath5/stats_aware_gans/aapm_generated_images/new/00035-128-auto1-gamma2-batch64-kimg000907 --seeds=0-10000 --network=/home/nkamath5/stats_aware_gans/training_runs_stats_aware_gans/00035-train_128-auto1-gamma2-kimg15000-batch64-noaug/network-snapshot-000907.pkl
#python generate.py --outdir=/home/nkamath5/stats_aware_gans/aapm_generated_images/new/00035-128-auto1-gamma2-batch64-kimg001814 --seeds=0-10000 --network=/home/nkamath5/stats_aware_gans/training_runs_stats_aware_gans/00035-train_128-auto1-gamma2-kimg15000-batch64-noaug/network-snapshot-001814.pkl
#python generate.py --outdir=/home/nkamath5/stats_aware_gans/aapm_generated_images/new/00035-128-auto1-gamma2-batch64-kimg002721 --seeds=0-10000 --network=/home/nkamath5/stats_aware_gans/training_runs_stats_aware_gans/00035-train_128-auto1-gamma2-kimg15000-batch64-noaug/network-snapshot-002721.pkl
#python generate.py --outdir=/home/nkamath5/stats_aware_gans/aapm_generated_images/new/00035-128-auto1-gamma2-batch64-kimg003628 --seeds=0-10000 --network=/home/nkamath5/stats_aware_gans/training_runs_stats_aware_gans/00035-train_128-auto1-gamma2-kimg15000-batch64-noaug/network-snapshot-003628.pkl
#python generate.py --outdir=/home/nkamath5/stats_aware_gans/aapm_generated_images/new/00035-128-auto1-gamma2-batch64-kimg004536 --seeds=0-10000 --network=/home/nkamath5/stats_aware_gans/training_runs_stats_aware_gans/00035-train_128-auto1-gamma2-kimg15000-batch64-noaug/network-snapshot-004536.pkl


#python generate.py --outdir=/home/nkamath5/stats_aware_gans/aapm_generated_images/new/00037-128-auto1-gamma2-batch128-kimg000921 --seeds=0-10000 --network=/home/nkamath5/stats_aware_gans/training_runs_stats_aware_gans/00037-train_128-auto1-gamma2-kimg15000-batch128-noaug/network-snapshot-000921.pkl
#python generate.py --outdir=/home/nkamath5/stats_aware_gans/aapm_generated_images/new/00037-128-auto1-gamma2-batch128-kimg001843 --seeds=0-10000 --network=/home/nkamath5/stats_aware_gans/training_runs_stats_aware_gans/00037-train_128-auto1-gamma2-kimg15000-batch128-noaug/network-snapshot-001843.pkl
#python generate.py --outdir=/home/nkamath5/stats_aware_gans/aapm_generated_images/new/00037-128-auto1-gamma2-batch128-kimg002764 --seeds=0-10000 --network=/home/nkamath5/stats_aware_gans/training_runs_stats_aware_gans/00037-train_128-auto1-gamma2-kimg15000-batch128-noaug/network-snapshot-002764.pkl
#python generate.py --outdir=/home/nkamath5/stats_aware_gans/aapm_generated_images/new/00037-128-auto1-gamma2-batch128-kimg003686 --seeds=0-10000 --network=/home/nkamath5/stats_aware_gans/training_runs_stats_aware_gans/00037-train_128-auto1-gamma2-kimg15000-batch128-noaug/network-snapshot-003686.pkl
#python generate.py --outdir=/home/nkamath5/stats_aware_gans/aapm_generated_images/new/00037-128-auto1-gamma2-batch128-kimg004608 --seeds=0-10000 --network=/home/nkamath5/stats_aware_gans/training_runs_stats_aware_gans/00037-train_128-auto1-gamma2-kimg15000-batch128-noaug/network-snapshot-004608.pkl
#python generate.py --outdir=/home/nkamath5/stats_aware_gans/aapm_generated_images/new/00037-128-auto1-gamma2-batch128-kimg005529 --seeds=0-10000 --network=/home/nkamath5/stats_aware_gans/training_runs_stats_aware_gans/00037-train_128-auto1-gamma2-kimg15000-batch128-noaug/network-snapshot-005529.pkl

#python generate.py --outdir=/home/nkamath5/stats_aware_gans/aapm_generated_images/new/00039-128-auto1-gamma2-batch128-abskimg006450 --seeds=0-10000 --network=/home/nkamath5/stats_aware_gans/training_runs_stats_aware_gans/00039-train_128-auto1-gamma2-kimg15000-batch128-noaug-resumecustom/network-snapshot-000921.pkl
#python generate.py --outdir=/home/nkamath5/stats_aware_gans/aapm_generated_images/new/00039-128-auto1-gamma2-batch128-abskimg007372 --seeds=0-10000 --network=/home/nkamath5/stats_aware_gans/training_runs_stats_aware_gans/00039-train_128-auto1-gamma2-kimg15000-batch128-noaug-resumecustom/network-snapshot-001843.pkl
#python generate.py --outdir=/home/nkamath5/stats_aware_gans/aapm_generated_images/new/00039-128-auto1-gamma2-batch128-abskimg008293 --seeds=0-10000 --network=/home/nkamath5/stats_aware_gans/training_runs_stats_aware_gans/00039-train_128-auto1-gamma2-kimg15000-batch128-noaug-resumecustom/network-snapshot-002764.pkl
#python generate.py --outdir=/home/nkamath5/stats_aware_gans/aapm_generated_images/new/00039-128-auto1-gamma2-batch128-abskimg009215 --seeds=0-10000 --network=/home/nkamath5/stats_aware_gans/training_runs_stats_aware_gans/00039-train_128-auto1-gamma2-kimg15000-batch128-noaug-resumecustom/network-snapshot-003686.pkl
#python generate.py --outdir=/home/nkamath5/stats_aware_gans/aapm_generated_images/new/00039-128-auto1-gamma2-batch128-abskimg010137 --seeds=0-10000 --network=/home/nkamath5/stats_aware_gans/training_runs_stats_aware_gans/00039-train_128-auto1-gamma2-kimg15000-batch128-noaug-resumecustom/network-snapshot-004608.pkl
#python generate.py --outdir=/home/nkamath5/stats_aware_gans/aapm_generated_images/new/00039-128-auto1-gamma2-batch128-abskimg011058 --seeds=0-10000 --network=/home/nkamath5/stats_aware_gans/training_runs_stats_aware_gans/00039-train_128-auto1-gamma2-kimg15000-batch128-noaug-resumecustom/network-snapshot-005529.pkl


#python generate.py --outdir=/home/nkamath5/stats_aware_gans/aapm_generated_images/new/00027-128-auto1-gamma2-batch64-kimg007257 --seeds=0-10000 --network=/home/nkamath5/stats_aware_gans/debug_training_runs_stats_aware_gans/00024-train_128-auto1-gamma2-kimg10000-batch64-noaug/network-snapshot-007257.pkl
#python generate.py --outdir=/home/nkamath5/stats_aware_gans/aapm_generated_images/new/00027-128-auto1-gamma2-batch64-kimg008164 --seeds=0-10000 --network=/home/nkamath5/stats_aware_gans/debug_training_runs_stats_aware_gans/00024-train_128-auto1-gamma2-kimg10000-batch64-noaug/network-snapshot-008164.pkl
#python generate.py --outdir=/home/nkamath5/stats_aware_gans/aapm_generated_images/new/00027-128-auto1-gamma2-batch64-kimg009072 --seeds=0-10000 --network=/home/nkamath5/stats_aware_gans/training_runs_stats_aware_gans/00027-train_128-auto1-gamma2-kimg10000-batch64-noaug/network-snapshot-009072.pkl
#python generate.py --outdir=/home/nkamath5/stats_aware_gans/aapm_generated_images/new/00027-128-auto1-gamma2-batch64-kimg009979 --seeds=0-10000 --network=/home/nkamath5/stats_aware_gans/training_runs_stats_aware_gans/00027-train_128-auto1-gamma2-kimg10000-batch64-noaug/network-snapshot-009979.pkl

#python generate.py --outdir=/home/nkamath5/stats_aware_gans/aapm_generated_images/00018-paper512-gamma100-batch48-kimg001512 --seeds=0-10000 --network=/home/nkamath5/stats_aware_gans/training_runs_stats_aware_gans/00018-challenge_data-paper512-gamma100-batch48-noaug/network-snapshot-001512.pkl
#python generate.py --outdir=/home/nkamath5/stats_aware_gans/aapm_generated_images/00021-paper512-gamma100-batch48-kimg000302 --seeds=0-10000 --network=/home/nkamath5/stats_aware_gans/training_runs_stats_aware_gans/00021-challenge_data-paper512-gamma100-batch48-noaug-resumecustom/network-snapshot-000302.pkl

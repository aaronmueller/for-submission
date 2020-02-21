#!/bin/bash
# training script created to work on CLSP grid (any user)

export LD_LIBRARY_PATH=/opt/NVIDIA/cuda-10/lib64
export CUDA_VISIBLE_DEVICES=`free-gpu`
source /home/amueller/miniconda3/bin/activate
conda activate parlai


cd /export/b10/amueller/discourse/hw3/discourse-hw3
python -u examples/train_model.py \
	-m internal:hred \
	-t internal:dailydialog \
	-bs 64 \
	--hiddensize 512 \
	--contextsize 512 \
	-eps 10 \
	-mf parlai_internal/zoo/movie_hred/hred_model2.ckpt \
	-vtim 10800 \
	-veps 1 \
	-sval True \
	-ttim 86400

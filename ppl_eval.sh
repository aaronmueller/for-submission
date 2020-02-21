#!/bin/bash

source /home/amueller/miniconda3/bin/activate
conda activate parlai

python parlai/scripts/eval_model.py \
	-mf parlai_internal/zoo/movie_hred/hred_model.ckpt.checkpoint \
	-m internal:hred \
	-t internal:dailydialog \
	-dt test \
	-mcs ppl \
	-rf ppl_eval

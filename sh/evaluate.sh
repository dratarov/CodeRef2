#!/bin/sh

# pre-training
python src/evaluate.py \
--dataset-root ../SPT-Code-Dataset/dataset \
--dataset-save-dir ../SPT-Code-Dataset/dataset/dataset_saved \
--vocab-save-dir ../SPT-Code-Dataset/dataset/vocab_saved_new \
--project-name test_eval \
--logging-file-path output \
--batch-size 32 \
--eval-batch-size 32 \
--n-epoch 5 \
--n-gpu 0 \
--fp16 \
--model-name pre_train

#--increase-token-embeddings True \

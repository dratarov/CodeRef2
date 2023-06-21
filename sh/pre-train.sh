#!/bin/sh

# pre-training
python src/pre_train.py \
--dataset-root ../SPT-Code-Dataset/dataset \
--dataset-save-dir ../SPT-Code-Dataset/dataset/dataset_saved \
--vocab-save-dir ../SPT-Code-Dataset/dataset/vocab_saved_new \
--do-fine-tune \
--increase-token-embeddings True \
--project-name rtd \
--logging-file-path output \
--do-pre-train \
--pre-train-tasks rtd \
--batch-size 16 \
--eval-batch-size 16 \
--n-epoch 5 \
--n-gpu 1 \
--fp16 \
--model-name pre_train

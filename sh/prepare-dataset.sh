#!/bin/sh

python src/prepare_dataset.py \
--dataset-root ../SPT-Code-Dataset/dataset \
--dataset-save-dir ../SPT-Code-Dataset/dataset/dataset_saved \
--logging-file-path output

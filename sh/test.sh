python src/test.py \
--dataset-root ../SPT-Code-Dataset/dataset \
--dataset-save-dir ../SPT-Code-Dataset/dataset/dataset_saved \
--vocab-save-dir ../SPT-Code-Dataset/dataset/vocab_saved_new \
--do-fine-tune \
--project-name rtd \
--logging-file-path output \
--do-pre-train \
--pre-train-tasks rtd \
--batch-size 32 \
--eval-batch-size 32 \
--n-epoch 5 \
--n-gpu 1 \
--fp16 \
--model-name pre_train
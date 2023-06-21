# CodeRef T5 Pre-training

This repository provides a framework for training the T5 model using different pre-training strategies. The documentation below will guide you through the necessary steps to set up and run the pre-training and fine-tunning processes.

## Before Pre-training

Before starting the pre-training process, there are a few steps you need to take:

1. **Generate Tokenizers**

   In order to tokenize your data, you need to run the `init_all()` function in the `src/init_tokenizers.py` script. After that, run the `load_and_test()` function to verify that the tokenization process is working correctly. This step ensures that your data can be properly processed during pre-training.

2. **Prepare Pre-training Dataset**

   The script `src/prepare_dataset.py` is provided to help you prepare your pre-training dataset. It is recommended to create a pickle file from your dataset, which will enable easy loading and quick sampling. Generating samples for the RTD (Reversible Text Deletion) task can be time-consuming, so creating a pickle file will help mitigate this.

3. **Preprocess Fine-tuning Dataset**

   To prepare your fine-tuning dataset, you should use the `src/pre_process.py` script. This script generates six dataset files:

   - `test.buggy-fixed.buggy`
   - `test.buggy-fixed.fixed`
   - `train.buggy-fixed.buggy`
   - `train.buggy-fixed.fixed`
   - `valid.buggy-fixed.buggy`
   - `valid.buggy-fixed.fixed`

   These files contain the necessary data for fine-tuning your T5 model.

## Installation and Pre-training

To install the required packages and run the pre-training process, follow these steps:

1. **Install Python 3.7 and related dependencies:**

   ```bash
   sudo apt install software-properties-common && sudo add-apt-repository ppa:deadsnakes/ppa && sudo apt install python3.7 && sudo apt-get install python3.7-venv && sudo apt-get install python3.7-dev
   ```

2. **Create a virtual environment:**

   ```bash
   python3.7 -m venv venv
   ```

3. **Activate the virtual environment:**

   ```bash
   . venv/bin/activate
   ```

4. **Install the required packages:** 

   ```bash
   pip install -r requirements.txt
   ```

5. **Log in to Weights & Biases (WandB):** 

   ```bash
   wandb login
   ```

6. **Modify the `sh/pre-train.sh` file:** 

   You will need to change the follwoing parameters:
   - `project-name`: Specify the name of your project.
   - `pre-train-tasks`: Choose the pre-training tasks you want to use (separated by comma).
   - `batch-size`: Set the batch size for training.
   - `eval-batch-size`: Set the batch size for evaluation.
   - `n-epochs`: Specify the number of training epochs.

7. **Grant execute permissions to the pre-training script:** 

   ```bash
   chmod +x ./pre-train.sh
   ```

8. **Start the pre-training process:** 

   ```bash
   nohup ./sh/pre-train.sh > /dev/null 2>&1&
   ```

9. **Monitor GPU usage:** 

   ```bash
   nvidia-smi
   ```

Note: If you encounter CUDA out of memory issues, try lowering the batch size parameters. Additionally, for tasks like cap, mng, and mdp, set increase-token-embeddings to TRUE. If the model stops improving during pre-training and produces null results for the loss function, you may need to re-run the model from the last checkpoint.

## Evaluation

To evaluate your trained model, you can use the `src/evaluate.py` script.

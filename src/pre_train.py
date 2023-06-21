"""
Reference: https://github.com/ramsrigouthamg/Paraphrase-any-question-with-T5-Text-To-Text-Transfer-Transformer-
Next ref: https://github.com/ceshine/finetuning-t5/tree/master/paraphrase
"""
import enum
import os
import sys
import argparse
import logging
from prettytable import PrettyTable
from args import add_args
from dataclasses import dataclass, asdict
import pytorch_lightning as pl
import pytorch_lightning_spells as pls
from pytorch_lightning.loggers import WandbLogger
from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config

import enums
from model.model import BaseConfig, T5BaseModel
from data.vocab import load_vocab
from data.dataset import init_dataset


logger = logging.getLogger(__name__)


class T5Model(T5BaseModel):
    def __init__(self, config: BaseConfig, args, mode='pre_training', bug_fix_size='small', **kwargs):
        model = T5ForConditionalGeneration.from_pretrained(config.base_t5_model)
        # init tokenizers
        code_vocab = load_vocab(vocab_root=args.vocab_root, name=args.code_vocab_name)
        ast_vocab = load_vocab(vocab_root=args.vocab_root, name=args.ast_vocab_name)
        nl_vocab = load_vocab(vocab_root=args.vocab_root, name=args.nl_vocab_name)
        # init dataset
        if mode == 'pre_training':
            dataset = init_dataset(args=args, mode=enums.TRAINING_MODE_PRE_TRAIN, load_if_saved=True)
        elif mode == 'fine_tunning':
            dataset = [
                init_dataset(args=args, mode=enums.TRAINING_MODE_FINE_TUNE, task=enums.TASK_BUG_FIX, language=bug_fix_size, split='train'),
                init_dataset(args=args, mode=enums.TRAINING_MODE_FINE_TUNE, task=enums.TASK_BUG_FIX, language=bug_fix_size, split='valid'),
                init_dataset(args=args, mode=enums.TRAINING_MODE_FINE_TUNE, task=enums.TASK_BUG_FIX, language=bug_fix_size, split='test')
            ]

        super().__init__(config, model, dataset, mode, args, code_vocab, nl_vocab, ast_vocab)

        self.config = config
        # log the config values
        self.save_hyperparameters(asdict(config))
        if mode == 'pre_training':
            print("Train dataset: ", len(self.dataset))
        elif mode == 'fine_tunning':
            print("Train dataset: ", len(self.train_dataset))


def main(args):
    pl.seed_everything(int(os.environ.get("SEED", 738)))

    config = BaseConfig(
        base_t5_model="t5-small",
        learning_rate=1e-4,
        epochs=args.n_epoch,
        grad_accu=1,
        batch_size=args.batch_size,
        fp16=args.fp16,
        num_gpus=args.n_gpu
    )

    wandb_logger = WandbLogger(project=args.project_name)
    wandb_logger.experiment.config["batch_size"] = args.batch_size

    previous_best_checkpoint = ''
    if args.pre_train_tasks:
        for i, task in enumerate(args.pre_train_tasks.split(',')):
            print("TASK: ", task)
            if not os.path.exists(os.path.join(args.dataset_root, 'model_checkpoints', f'task_{task}')):
                os.makedirs(os.path.join(args.dataset_root, 'model_checkpoints', f'task_{task}'))
            callbacks = [
                pl.callbacks.ModelCheckpoint(
                    dirpath=os.path.join(args.dataset_root, 'model_checkpoints', f'task_{task}'),
                    monitor=None,
                    save_top_k=1,
                    save_last=True
                )
            ]
            if (i == 0):
                pl_module = T5Model(config, args)
            else:
                pl_module = T5Model.load_from_checkpoint(
                    previous_best_checkpoint,
                    config=config,
                    args=args
                )

            pl_module.set_task(task)
            trainer = pl.Trainer(
                default_root_dir=os.path.join(args.dataset_root, 'model_checkpoints', f'task_{task}'),
                accelerator='dp' if config.num_gpus > 1 else None,
                # amp_backend="apex", amp_level='O2',
                precision=16 if config.fp16 else 32,
                gpus=config.num_gpus,
                val_check_interval=1.0,
                # gradient_clip_val=10,
                max_epochs=config.epochs,
                # max_steps=steps,
                callbacks=callbacks,
                accumulate_grad_batches=config.grad_accu,
                # auto_scale_batch_size='power' if batch_size is None else None,
                logger=wandb_logger,
                log_every_n_steps=1000
            )

            trainer.fit(pl_module)

            assert isinstance(callbacks[0], pl.callbacks.ModelCheckpoint)
            print("Current best model path: ", callbacks[0].best_model_path)
            previous_best_checkpoint = callbacks[0].best_model_path
            pl_module = T5Model.load_from_checkpoint(
                callbacks[0].best_model_path,
                config=config,
                args=args
            )

        pl_module.model.save_pretrained(os.path.join(args.dataset_root, f"{config.base_t5_model}_best"))
        print("Best model saved")
    
    if args.do_fine_tune:
        # there was a pre-training phase before
        if args.pre_train_tasks:
            print('loading from previous pre-training tasks...')
            pl_module = T5Model.load_from_checkpoint(
                previous_best_checkpoint,
                config=config,
                args=args,
                mode='fine_tunning',
                bug_fix_size='small'
            )
        else:
            print('loading from scratch...')
            pl_module = T5Model(
                config=config,
                args=args,
                mode='fine_tunning',
                bug_fix_size='small'
            )

        # FOR SMALL
        task = 'bug_fix'
        print("TASK: ", task + ' small')
        if not os.path.exists(os.path.join(args.dataset_root, 'model_checkpoints', f'task_{task}_small')):
            os.makedirs(os.path.join(args.dataset_root, 'model_checkpoints', f'task_{task}_small'))
        callbacks = [
            pl.callbacks.ModelCheckpoint(
                dirpath=os.path.join(args.dataset_root, 'model_checkpoints', f'task_{task}_small'),
                monitor=None,
                save_top_k=1,
                save_last=True
            )
        ]

        pl_module.set_task(task)
        trainer = pl.Trainer(
            default_root_dir=os.path.join(args.dataset_root, 'model_checkpoints', f'task_{task}_small'),
            accelerator='dp' if config.num_gpus > 1 else None,
            # amp_backend="apex", amp_level='O2',
            precision=16 if config.fp16 else 32,
            gpus=config.num_gpus,
            val_check_interval=1.0,
            # gradient_clip_val=10,
            max_epochs=10,
            # max_steps=steps,
            callbacks=callbacks,
            accumulate_grad_batches=config.grad_accu,
            # auto_scale_batch_size='power' if batch_size is None else None,
            logger=wandb_logger,
            log_every_n_steps=1000
        )

        trainer.fit(pl_module)
        assert isinstance(callbacks[0], pl.callbacks.ModelCheckpoint)
        print("Current best model path: ", callbacks[0].best_model_path)
        previous_best_checkpoint = callbacks[0].best_model_path
        pl_module = T5Model.load_from_checkpoint(
            callbacks[0].best_model_path,
            config=config,
            args=args,
            mode='fine_tunning',
            bug_fix_size='small'
        )
        print('Testing for small bug fix dataset...')
        trainer.test(pl_module)

        pl_module.model.save_pretrained(os.path.join(args.dataset_root, f"{config.base_t5_model}_best_small_bug_fix"))
        print("Best model saved for small bug fix")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.register('type', 'bool', lambda v: v.lower() in ['yes', 'true', 't', '1', 'y'])

    add_args(parser)

    main_args = parser.parse_args()
    main_args.vocab_root = main_args.vocab_save_dir

    if not os.path.exists(main_args.logging_file_path):
        os.makedirs(main_args.logging_file_path)

    # logging, log to both console and file, debug level only to file
    logger = logging.getLogger()
    logger.setLevel(level=logging.DEBUG)

    console = logging.StreamHandler()
    console.setLevel(level=logging.INFO)
    logger.addHandler(console)

    file = logging.FileHandler(os.path.join(main_args.logging_file_path, 'info.log'))
    file.setLevel(level=logging.INFO)
    formatter = logging.Formatter('[%(asctime)s | %(filename)s | line %(lineno)d] - %(levelname)s: %(message)s')
    file.setFormatter(formatter)
    logger.addHandler(file)

    # log command and configs
    logger.debug('COMMAND: {}'.format(' '.join(sys.argv)))

    config_table = PrettyTable()
    config_table.field_names = ["Configuration", "Value"]
    config_table.align["Configuration"] = "l"
    config_table.align["Value"] = "l"
    for config, value in vars(main_args).items():
        config_table.add_row([config, str(value)])
    logger.debug('Configurations:\n{}'.format(config_table))

    main(main_args)

import math
from functools import partial
from dataclasses import dataclass
from typing import Callable, List, Union

import torch
import numpy as np
import pytorch_lightning as pl
import pytorch_lightning_spells as pls
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from transformers import T5ForConditionalGeneration
from transformers import get_linear_schedule_with_warmup, AdamW

from data.vocab import Vocab
from data.dataset import CodeDataset
from data.data_collator import collate_fn
from eval.metrics import bleu


@dataclass
class BaseConfig:
    base_t5_model: str
    batch_size: int
    fp16: bool
    learning_rate: float
    epochs: int
    num_gpus: int = 1
    grad_accu: int = 1
    tpu_cores: int = 0


class T5BaseModel(pl.LightningModule):
    def __init__(self, config: BaseConfig, model: T5ForConditionalGeneration, dataset: Union[CodeDataset, List[CodeDataset]], mode: str, args, code_vocab: Vocab, nl_vocab: Vocab, ast_vocab: Vocab, **kwargs):
        super().__init__()
        self.config = config
        self.args = args
        self.mode = mode # pre_training or fine_tunning

        #tokenizers
        self.code_vocab = code_vocab
        self.nl_vocab = nl_vocab
        self.ast_vocab = ast_vocab

        # dataset
        if self.mode == 'pre_training':
            self.dataset: CodeDataset = dataset
        elif self.mode == 'fine_tunning':
            assert (type(dataset) == list) and (len(dataset) == 3)
            self.train_dataset: CodeDataset = dataset[0]
            self.valid_dataset: CodeDataset = dataset[1]
            self.test_dataset: CodeDataset = dataset[2]
        
        #########
        # if self.mode == 'pre_training':
        #     nk = 2
        #     self.pk = .5
        #     self.dataset.languages =  self.dataset.languages[:nk]
        #     self.dataset.sources = self.dataset.sources[:nk]
        #     self.dataset.codes = self.dataset.codes[:nk]
        #     self.dataset.asts = self.dataset.asts[:nk]
        #     self.dataset.names = self.dataset.names[:nk]
        #     self.dataset.codes_wo_name = self.dataset.codes_wo_name[:nk]
        #     self.dataset.names_wo_name = self.dataset.names_wo_name[:nk]
        #     self.dataset.only_names = self.dataset.only_names[:nk]
        #     self.dataset.docs = self.dataset.docs[:nk]
        #     self.dataset.rtd_masked_code = self.dataset.rtd_masked_code[:nk] 
        #     self.dataset.rtd_output = self.dataset.rtd_output[:nk]
        #     self.dataset.size = nk
        # elif self.mode == 'fine_tunning':
        #     nk = 2
        #     self.train_dataset.codes = self.train_dataset.codes[:nk]
        #     self.train_dataset.targets = self.train_dataset.targets[:nk]
        #     self.train_dataset.size = nk

        #     self.valid_dataset.codes = self.valid_dataset.codes[:nk]
        #     self.valid_dataset.targets = self.valid_dataset.targets[:nk]
        #     self.valid_dataset.size = nk

        #     self.test_dataset.codes = self.test_dataset.codes[:nk]
        #     self.test_dataset.targets = self.test_dataset.targets[:nk]
        #     self.test_dataset.size = nk
        #########

        if self.mode == 'pre_training':
            dataset_size = len(dataset)
            indices = list(range(dataset_size))
            split = int(np.floor(.1 * dataset_size))
            train_indices, val_indices = indices[split:], indices[:split]
            self.train_sampler = SubsetRandomSampler(train_indices)
            self.valid_sampler = SubsetRandomSampler(val_indices)
        elif self.mode == 'fine_tunning':
            pass

        # the actual stuffs
        self.model = model
        self.task = None

        # tokenizer
        if self.args.increase_token_embeddings:
            self.model.config.vocab_size = 100000 # 100,000 for cap/mng/mdp tasks
            self.model.resize_token_embeddings(100000) # 100,000 for cap/mng/mdp tasks
        else:
            self.model.config.vocab_size = len(code_vocab) # 100,000 for cap/mng/mdp tasks
            self.model.resize_token_embeddings(len(code_vocab)) # 100,000 for cap/mng/mdp tasks

        self.collate_fn = partial(
            collate_fn,
            args=self.args,
            code_vocab=self.code_vocab,
            nl_vocab=self.nl_vocab,
            ast_vocab=self.ast_vocab
        )

        self.em = 0

        self.train_loss_tracker = pls.utils.EMATracker(alpha=0.02)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs.loss, outputs.logits

    def train_dataloader(self):
        if self.mode == 'pre_training':
            train_collate_fn = partial(
                self.collate_fn,
                dataset=self.dataset,
            )
            return DataLoader(self.dataset, batch_size=self.args.batch_size, num_workers=8, drop_last=True, collate_fn=train_collate_fn, sampler=self.train_sampler)
        elif self.mode == 'fine_tunning':
            train_collate_fn = partial(
                self.collate_fn,
                dataset=self.train_dataset,
            )
            return DataLoader(self.train_dataset, batch_size=self.args.batch_size, num_workers=8, drop_last=True, collate_fn=train_collate_fn)

    def val_dataloader(self):
        if self.mode == 'pre_training':
            valid_collate_fn = partial(
                self.collate_fn,
                dataset=self.dataset,
            )
            return DataLoader(self.dataset, batch_size=self.args.eval_batch_size, num_workers=8, drop_last=True, collate_fn=valid_collate_fn, sampler=self.valid_sampler)
        elif self.mode == 'fine_tunning':
            valid_collate_fn = partial(
                self.collate_fn,
                dataset=self.valid_dataset,
            )
            return DataLoader(self.valid_dataset, batch_size=self.args.eval_batch_size, num_workers=8, drop_last=True, collate_fn=valid_collate_fn)

    def test_dataloader(self):
        if self.mode == 'fine_tunning':
            test_collate_fn = partial(
                self.collate_fn,
                dataset=self.test_dataset,
            )
            return DataLoader(self.test_dataset, batch_size=self.args.eval_batch_size, num_workers=8, drop_last=True, collate_fn=test_collate_fn)

    def set_task(self, task):
        self.task = task
        if self.mode == 'pre_training':
            self.dataset.set_task(task)
        elif self.mode == 'fine_tunning':
            self.train_dataset.set_task(task)
            self.valid_dataset.set_task(task)
            self.test_dataset.set_task(task)

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        loss, output = self(input_ids, attention_mask, labels)
        self.log("val_loss", loss)
        return { 'loss': loss }

    def validation_epoch_end(self, outputs):
        loss = sum([o['loss'] for o in outputs]) / len(outputs)
        self.log("val_loss_average", loss)

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        loss, output = self(input_ids, attention_mask, labels)

        self.log("train_loss", loss)
        return { 'loss': loss }

    def training_epoch_end(self, outputs):
        loss = sum([o['loss'] for o in outputs]) / len(outputs)
        self.log("train_loss_average", loss)

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        loss, output = self(input_ids, attention_mask, labels)
        beam_output = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_beams=5,
            early_stopping=True
        )

        predictions_decoded = self.code_vocab.decode_batch(beam_output.tolist())

        labels = labels.tolist()
        for label in labels:
            for i, elem in enumerate(label):
                if elem == -100:
                    label[i] = 0
        labels_decoded = self.code_vocab.decode_batch(labels)
        refs = [ref.strip().split() for ref in labels_decoded]
        cans = [can.strip().split() for can in predictions_decoded]

        bleu_score = bleu(references=refs, candidates=cans)
        print('BLEU: ', bleu_score)

        # calc em
        idx = 0
        idxs = []
        for r, c in zip(refs, cans):
            if r == c:
                self.em += 1
                idxs.append(16*batch_idx + idx)
            idx += 1
        print('EM: ', self.em)
        print()
        print('IDXS: ', idxs)
        print()

        metrics = { "test_loss": loss, "em": self.em, "bleu": bleu_score['bleu'] }
        self.log_dict(metrics)
        return metrics

    def test_epoch_end(self, outputs):
        loss = sum([o['test_loss'] for o in outputs]) / len(outputs)
        bleu_score = sum([o['bleu'] for o in outputs]) / len(outputs)
        metrics = { "avg_test_loss": loss, "em": self.em, "average_test_bleu": bleu_score }
        self.log_dict(metrics)

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.config.learning_rate)
        if self.mode == 'pre_training':
            train_dataset_length = int(np.floor(.9*len(self.dataset)))
        elif self.mode == 'fine_tunning':
            train_dataset_length = len(self.train_dataset)
        steps_per_epochs = math.floor(
            train_dataset_length / self.config.batch_size / self.config.grad_accu  # / self.num_gpus # dpp mode
        )
        n_steps = steps_per_epochs * self.config.epochs
        scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=0,
                num_training_steps=n_steps)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

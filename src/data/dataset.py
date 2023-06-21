import torch.utils.data
from torch.utils.data.dataset import Dataset

import os
import random
import logging
import pickle
import random

import enums
from .data_utils import load_dataset_from_dir, parse_for_bug_fix
from .vocab import Vocab
from eval.bleu.google_bleu import avg_bleu

from generator_network import generator_java

logger = logging.getLogger(__name__)


class CodeDataset(Dataset):

    def __init__(self, args, dataset_name, mode, task=None, language=None, split=None, clone_mapping=None):
        """
        Initialization definition.

        Args:
            args (argparse.Namespace): Arguments
            dataset_name (str): Name of the dataset
            mode (str): Training mode, ``pre_train`` or ``fine_tune``
            task (str): Dataset mode, support pre-training tasks: ['cap', 'mass', 'mnp'],
                and downstream fine-tuning task: ['summarization', 'translation'],
                future support ['completion', 'search', 'clone', 'summarization', 'translation']
            language (str): Only for downstream fine-tuning
            split (str): Only for downstream fine-tuning, support ['train', 'valid', 'test', 'codebase']
            clone_mapping (dict[int, str]): Mapping from code id to source code string, use only for clone detection
        """
        super(CodeDataset, self).__init__()
        self.args = args
        self.dataset_name = dataset_name
        self.task = task
        self.mode = mode
        self.split = split
        self.paths = {}

        # dataset dir for files, all files in this dir meeting the filename will be used as dataset files
        self.dataset_dir = os.path.join(args.dataset_root, self.mode)

        # load pre-training dataset
        if self.mode == 'pre_train':
            self.paths, self.languages, self.sources, self.codes, self.asts, self.names, self.codes_wo_name, \
                self.names_wo_name, self.only_names, self.docs, self.rtd_masked_code, self.rtd_output, self.code_tags = load_dataset_from_dir(dataset_dir=self.dataset_dir)
            self.size = len(self.codes)
        # load fine-tuning dataset
        else:
            assert split
            logger.info(f'  Loading {split} set')
            self.dataset_dir = os.path.join(self.dataset_dir, task)
            # bug fix
            if task == enums.TASK_BUG_FIX:
                assert split in ['train', 'valid', 'test']
                # language here stands for dataset scale
                assert language in ['small', 'medium']
                self.dataset_dir = os.path.join(self.dataset_dir, language)
                buggy_path = os.path.join(self.dataset_dir, f'{split}.buggy-fixed.buggy')
                fixed_path = os.path.join(self.dataset_dir, f'{split}.buggy-fixed.fixed')
                self.paths['buggy'] = buggy_path
                self.paths['fixed'] = fixed_path
                self.codes, self.comments, self.targets = parse_for_bug_fix(
                    buggy_path=buggy_path,
                    fixed_path=fixed_path
                )
                assert len(self.codes) == len(self.targets) == len(self.comments)
                self.size = len(self.codes)

    def __getitem__(self, index):
        # rtd
        if self.task == enums.TASK_RTD:
            task_prefix = 'TASK_RTD: '

            return task_prefix + self.rtd_masked_code[index], self.rtd_output[index]

        # atc
        elif self.task == enums.TASK_AUTOCOMPLETION:
            task_prefix = 'TASK_AUTOCOMPLETION: '

            code = self.codes[index].split()
            context_len = int(random.randint(20, 90) / 100 * len(code))
            if (not context_len):
                context_len = 1
            context = code[:context_len]
            res = code[context_len:]
            return task_prefix + ' '.join(context), ' '.join(res)

        # mass
        elif self.task == enums.TASK_MASS:
            task_prefix = 'TASK_MASS: '

            code_tokens = self.codes[index].split()
            mask_len = int(self.args.mass_mask_ratio * len(code_tokens))
            mask_start = random.randint(0, len(code_tokens) - mask_len)
            mask_tokens = code_tokens[mask_start: mask_start + mask_len]
            input_tokens = code_tokens[:mask_start] + [Vocab.MSK_TOKEN] + code_tokens[mask_start + mask_len:]
            return task_prefix + ' '.join(input_tokens), ' '.join(mask_tokens)

        # mlm
        elif self.task == enums.TASK_MLM:
            task_prefix = 'TASK_MLM: '

            code_tokens = self.codes[index].split()
            input_tokens = [Vocab.MSK_TOKEN if random.random() <= 0.15 else token for token in code_tokens]
            return task_prefix + ' '.join(input_tokens), ' '.join(code_tokens)

        # cap
        elif self.task == enums.TASK_CODE_AST_PREDICTION:
            task_prefix = 'TASK_CODE_AST_PREDICTION: '

            is_ast = random.random() < 0.5
            if is_ast:
                return task_prefix + self.codes[index] + Vocab.SEP_TOKEN + self.asts[index], "1"
            else:
                other_ast = self.asts[random.randint(0, self.size - 1)]
                while other_ast == self.asts[index]:
                    other_ast = self.asts[random.randint(0, self.size - 1)]
                return task_prefix + self.codes[index] + Vocab.SEP_TOKEN + other_ast, "0"

        # mnp
        elif self.task == enums.TASK_METHOD_NAME_PREDICTION:
            task_prefix = 'TASK_METHOD_NAME_PREDICTION: '

            return task_prefix + self.codes_wo_name[index], self.only_names[index]

        # mdp
        elif self.task == enums.TASK_METHOD_DOCS_PREDICTION:
            task_prefix = 'TASK_METHOD_DOCS_PREDICTION: '

            return task_prefix + self.codes[index], self.docs[index]

        # it
        elif self.task == enums.TASK_IDENTIFIER_TAGGING:
            task_prefix = 'TASK_IDENTIFIER_TAGGING: '

            return task_prefix + self.codes[index], self.code_tags[index]

        # mip
        elif self.task == enums.TASK_MIP:
            task_prefix = 'TASK_MIP: '

            input_tokens = []
            for token, is_identifier in zip(self.codes[index].split(), self.code_tags[index].split()):
                if int(is_identifier):
                    input_tokens.append(Vocab.MSK_TOKEN)
                else:
                    input_tokens.append(token)
            return task_prefix + ' '.join(input_tokens), self.codes[index]

        # bug fix
        elif self.task == enums.TASK_BUG_FIX:
            task_prefix = 'TASK_BUG_FIX: '

            return task_prefix + self.codes[index] + Vocab.SEP_TOKEN + self.comments[index], self.targets[index]

    def __len__(self):
        return self.size

    def set_task(self, task):
        self.task = task

    def save(self):
        """Save to binary pickle file"""
        path = os.path.join(self.args.dataset_save_dir, f'{self.dataset_name}.pk')
        with open(path, mode='wb') as f:
            pickle.dump(self, f)
        logger.info(f'Dataset saved to {path}')

    def subset(self, ratio):
        """
        Return a subset of self.

        Args:
            ratio (float): The ratio of size, must greater than 0 and less than/equal to 1

        Returns:
            Dataset: the subset

        """
        assert 0 < ratio <= 1, f'The subset ratio supposed to be 0 < ratio <= 1, but got ratio={ratio}'
        if ratio == 1:
            return self
        indices = random.sample(range(self.size), int(self.size * ratio))
        return torch.utils.data.Subset(self, indices)


def init_dataset(args, mode, task=None, language=None, split=None, clone_mapping=None,
                 load_if_saved=True) -> CodeDataset:
    """
    Find dataset, if the dataset is saved, load and return, else initialize and return.

    Args:
        args (argparse.Namespace): Arguments
        mode (str): Training mode, ``pre_train`` or ``fine_tune``
        task (str): Dataset mode, support pre-training tasks: ['cap', 'mass', 'mnp'],
            and downstream fine-tuning task: ['summarization', 'translation'],
            future support ['completion', 'search', 'clone', 'summarization', 'translation']
        language (str): Only for downstream fine-tuning
        split (str): Only for downstream fine-tuning, support ['train', 'valid', 'test', 'codebase(only for search)']
        clone_mapping (dict[int, str]): Mapping from code id to source code string, use only for clone detection
        load_if_saved (bool): Whether to load the saved instance if it exists, default to True

    Returns:
        CodeDataset: Loaded or initialized dataset

    """
    name = '.'.join([sub_name for sub_name in [mode, task, language, split] if sub_name is not None])
    if load_if_saved:
        path = os.path.join(args.dataset_save_dir, f'{name}.pk')
        if os.path.exists(path) and os.path.isfile(path):
            logger.info(f'Trying to load saved binary pickle file from: {path}')
            with open(path, mode='rb') as f:
                obj = pickle.load(f)
            assert isinstance(obj, CodeDataset)
            obj.args = args
            logger.info(f'Dataset instance loaded from: {path}')
            print_paths(obj.paths)
            return obj
    dataset = CodeDataset(args=args,
                          dataset_name=name,
                          mode=mode,
                          task=task,
                          language=language,
                          split=split,
                          clone_mapping=clone_mapping)
    dataset.save()
    return dataset


def print_paths(paths):
    """
    Print paths.

    Args:
        paths (dict): Dict mapping path group to path string or list of path strings.

    """
    logger.info('Dataset loaded from these files:')
    for key, value in paths.items():
        if isinstance(value, list):
            for v in value:
                logger.info(f'  {key}: {v}')
        else:
            logger.info(f'  {key}: {value}')

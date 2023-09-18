import torch

from typing import List
import itertools

from data.vocab import Vocab
import enums


def collate_fn(batch, args, dataset, code_vocab, nl_vocab, ast_vocab):
    """
    Data collator function.

    Args:
        batch (list):
        args (argparse.Namespace):
        task (str):
        code_vocab (Vocab):
        nl_vocab (Vocab):
        ast_vocab (Vocab):

    Returns:
        dict: Model inputs

    """
    model_inputs = {}
    # rtd
    if dataset.task == enums.TASK_RTD:
        inputs, outputs = map(list, zip(*batch))

        model_inputs['input_ids'], model_inputs['attention_mask'] = get_batch_inputs(
            batch=inputs,
            vocab=code_vocab,
            processor=None,
            max_len=args.max_code_len
        )

        # model_inputs['decoder_input_ids'], model_inputs['decoder_attention_mask'] = get_batch_inputs(
        #     batch=outputs,
        #     vocab=code_vocab,
        #     processor=Vocab.sos_processor,
        #     max_len=args.max_code_len
        # )

        model_inputs['labels'], _ = get_batch_inputs(
            batch=outputs,
            vocab=code_vocab,
            processor=Vocab.eos_processor,
            max_len=args.max_code_len,
            is_label=True
        )

    # atc
    elif dataset.task == enums.TASK_AUTOCOMPLETION:
        inputs, outputs = map(list, zip(*batch))

        model_inputs['input_ids'], model_inputs['attention_mask'] = get_batch_inputs(
            batch=inputs,
            vocab=code_vocab,
            processor=None,
            max_len=args.max_code_len
        )

        # model_inputs['decoder_input_ids'], model_inputs['decoder_attention_mask'] = get_batch_inputs(
        #     batch=outputs,
        #     vocab=code_vocab,
        #     processor=Vocab.sos_processor,
        #     max_len=args.max_code_len
        # )

        model_inputs['labels'], _ = get_batch_inputs(
            batch=outputs,
            vocab=code_vocab,
            processor=Vocab.eos_processor,
            max_len=args.max_code_len,
            is_label=True
        )

    # cap
    elif dataset.task == enums.TASK_CODE_AST_PREDICTION:
        inputs, outputs = map(list, zip(*batch))


        model_inputs['input_ids'], model_inputs['attention_mask'] = get_batch_inputs_cap(
            batch=inputs,
            code_vocab=code_vocab,
            ast_vocab=ast_vocab,
            max_code_len=args.max_code_len,
            max_ast_len=args.max_ast_len
        )

        # model_inputs['decoder_input_ids'], model_inputs['decoder_attention_mask'] = get_batch_inputs(
        #     batch=outputs,
        #     vocab=code_vocab,
        #     processor=Vocab.sos_processor,
        #     max_len=args.max_code_len
        # )

        model_inputs['labels'], _ = get_batch_inputs(
            batch=outputs,
            vocab=code_vocab,
            processor=Vocab.eos_processor,
            max_len=args.max_code_len,
            is_label=True
        )

    # mnp
    elif dataset.task == enums.TASK_METHOD_NAME_PREDICTION:
        inputs, outputs = map(list, zip(*batch))

        model_inputs['input_ids'], model_inputs['attention_mask'] = get_batch_inputs(
            batch=inputs,
            vocab=code_vocab,
            processor=None,
            max_len=args.max_code_len
        )

        # model_inputs['decoder_input_ids'], model_inputs['decoder_attention_mask'] = get_batch_inputs(
        #     batch=outputs,
        #     vocab=nl_vocab,
        #     processor=Vocab.sos_processor,
        #     max_len=args.max_nl_len
        # )

        model_inputs['labels'], _ = get_batch_inputs(
            batch=outputs,
            vocab=nl_vocab,
            processor=Vocab.eos_processor,
            max_len=args.max_nl_len,
            is_label=True
        )

    # mass
    elif dataset.task == enums.TASK_MASS:
        inputs, outputs = map(list, zip(*batch))

        model_inputs['input_ids'], model_inputs['attention_mask'] = get_batch_inputs(
            batch=inputs,
            vocab=code_vocab,
            processor=None,
            max_len=args.max_code_len
        )

        # model_inputs['decoder_input_ids'], model_inputs['decoder_attention_mask'] = get_batch_inputs(
        #     batch=outputs,
        #     vocab=code_vocab,
        #     processor=Vocab.sos_processor,
        #     max_len=args.max_code_len
        # )

        model_inputs['labels'], _ = get_batch_inputs(
            batch=outputs,
            vocab=code_vocab,
            processor=Vocab.eos_processor,
            max_len=args.max_code_len,
            is_label=True
        )

    # mlm
    elif dataset.task == enums.TASK_MLM:
        inputs, outputs = map(list, zip(*batch))

        model_inputs['input_ids'], model_inputs['attention_mask'] = get_batch_inputs(
            batch=inputs,
            vocab=code_vocab,
            processor=None,
            max_len=args.max_code_len
        )

        # model_inputs['decoder_input_ids'], model_inputs['decoder_attention_mask'] = get_batch_inputs(
        #     batch=outputs,
        #     vocab=code_vocab,
        #     processor=Vocab.sos_processor,
        #     max_len=args.max_code_len
        # )

        model_inputs['labels'], _ = get_batch_inputs(
            batch=outputs,
            vocab=code_vocab,
            processor=Vocab.eos_processor,
            max_len=args.max_code_len,
            is_label=True
        )

    # mip
    elif dataset.task == enums.TASK_MIP:
        inputs, outputs = map(list, zip(*batch))

        model_inputs['input_ids'], model_inputs['attention_mask'] = get_batch_inputs(
            batch=inputs,
            vocab=code_vocab,
            processor=None,
            max_len=args.max_code_len
        )

        # model_inputs['decoder_input_ids'], model_inputs['decoder_attention_mask'] = get_batch_inputs(
        #     batch=outputs,
        #     vocab=code_vocab,
        #     processor=Vocab.sos_processor,
        #     max_len=args.max_code_len
        # )

        model_inputs['labels'], _ = get_batch_inputs(
            batch=outputs,
            vocab=code_vocab,
            processor=Vocab.eos_processor,
            max_len=args.max_code_len,
            is_label=True
        )

    # mdp
    elif dataset.task == enums.TASK_METHOD_DOCS_PREDICTION:
        inputs, outputs = map(list, zip(*batch))

        model_inputs['input_ids'], model_inputs['attention_mask'] = get_batch_inputs(
            batch=inputs,
            vocab=code_vocab,
            processor=None,
            max_len=args.max_code_len
        )

        # model_inputs['decoder_input_ids'], model_inputs['decoder_attention_mask'] = get_batch_inputs(
        #     batch=outputs,
        #     vocab=nl_vocab,
        #     processor=Vocab.sos_processor,
        #     max_len=args.max_nl_len
        # )

        model_inputs['labels'], _ = get_batch_inputs(
            batch=outputs,
            vocab=nl_vocab,
            processor=Vocab.eos_processor,
            max_len=args.max_nl_len,
            is_label=True
        )

    # it
    elif dataset.task == enums.TASK_IDENTIFIER_TAGGING:
        inputs, outputs = map(list, zip(*batch))

        model_inputs['input_ids'], model_inputs['attention_mask'] = get_batch_inputs(
            batch=inputs,
            vocab=code_vocab,
            processor=None,
            max_len=args.max_code_len
        )

        # model_inputs['decoder_input_ids'], model_inputs['decoder_attention_mask'] = get_batch_inputs(
        #     batch=outputs,
        #     vocab=code_vocab,
        #     processor=Vocab.sos_processor,
        #     max_len=args.max_code_len
        # )

        model_inputs['labels'], _ = get_batch_inputs(
            batch=outputs,
            vocab=code_vocab,
            processor=Vocab.eos_processor,
            max_len=args.max_code_len,
            is_label=True
        )

    # bug fix
    elif dataset.task in [enums.TASK_BUG_FIX, enums.CODE_2_CODE, enums.CODE_2_COD]:
        inputs, outputs = map(list, zip(*batch))
        max_code_len = 65 if args.bug_fix_scale == 'small' else 512

        model_inputs['input_ids'], model_inputs['attention_mask'] = get_batch_inputs_bug_fix(
            batch=inputs,
            code_vocab=code_vocab,
            nl_vocab=nl_vocab,
            max_code_len=max_code_len,
            max_comment_len=512
        )

        model_inputs['labels'], _ = get_batch_inputs(
            batch=outputs,
            vocab=code_vocab,
            processor=Vocab.eos_processor,
            max_len=max_code_len,
            is_label=True
        )

    return model_inputs


def get_batch_inputs(batch: List[str], vocab: Vocab, processor=None, max_len=None, is_label=False):
    """
    Encode the given batch to input to the model.

    Args:
        batch (list[str]): Batch of sequence,
            each sequence is represented by a string or list of tokens
        vocab (Vocab): Vocab of the batch
        processor (tokenizers.processors.PostProcessor): Optional, post-processor method
        max_len (int): Optional, the maximum length of each sequence

    Returns:
        (torch.LongTensor, torch.LongTensor): Tensor of batch and mask, [B, T]

    """
    # set post processor
    if processor:
        vocab.tokenizer.post_processor = processor
    # set truncation
    if max_len:
        vocab.tokenizer.enable_truncation(max_length=max_len)
    else:
        vocab.tokenizer.no_truncation()
    # encode batch
    inputs, padding_mask = vocab.encode_batch(batch, pad=True, max_length=max_len)
    if is_label:
        targets = inputs.copy()
        for i, target in enumerate(targets):
            for j, value in enumerate(target):
                if value == vocab.pad_token_id:
                    inputs[i][j] = -100
    # to tensor
    inputs = torch.tensor(inputs, dtype=torch.long)
    padding_mask = torch.tensor(padding_mask, dtype=torch.long)

    return inputs, padding_mask


def get_batch_inputs_cap(batch: List[str], code_vocab: Vocab, ast_vocab: Vocab, max_code_len=None, max_ast_len=None):
    sep_batch = [inputs.split('[SEP]') for inputs in batch]
    code_batch = [x[0] for x in sep_batch]
    ast_batch = [x[1] for x in sep_batch]

    code_inputs, code_padding_mask = get_batch_inputs(
        batch=code_batch,
        vocab=code_vocab,
        processor=Vocab.sep_processor,
        max_len=max_code_len
    )

    ast_inputs, ast_padding_mask = get_batch_inputs(
        batch=ast_batch,
        vocab=ast_vocab,
        processor=None,
        max_len=max_ast_len
    )

    inputs = torch.cat([inputs for inputs in [code_inputs, ast_inputs] if inputs is not None], dim=-1)
    padding_mask = torch.cat([mask for mask in [code_padding_mask, ast_padding_mask]
                              if mask is not None], dim=-1)

    return inputs, padding_mask


def get_batch_inputs_bug_fix(batch: List[str], code_vocab: Vocab, nl_vocab: Vocab, max_code_len=None, max_comment_len=None):
    sep_batch = [inputs.split('[SEP]') for inputs in batch]
    code_batch = [x[0] for x in sep_batch]
    comment_batch = [x[1] for x in sep_batch]

    code_inputs, code_padding_mask = get_batch_inputs(
        batch=code_batch,
        vocab=code_vocab,
        processor=Vocab.sep_processor,
        max_len=max_code_len
    )

    comment_inputs, comment_padding_mask = get_batch_inputs(
        batch=comment_batch,
        vocab=nl_vocab,
        processor=None,
        max_len=max_comment_len
    )

    inputs = torch.cat([inputs for inputs in [code_inputs, comment_inputs] if inputs is not None], dim=-1)
    padding_mask = torch.cat([mask for mask in [code_padding_mask, comment_padding_mask]
                              if mask is not None], dim=-1)

    return inputs, padding_mask

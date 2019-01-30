import os
import logging
import pickle
import torch
import torch.nn as nn
import sys
import preprocess

from collections import defaultdict
from torch.serialization import default_restore_location


def load_embedding(embed_path, dictionary):
    """Parse an embedding text file into an torch.nn.Embedding layer."""
    embed_dict, embed_dim = {}, None
    with open(embed_path) as file:
        embed_dim = int(next(file).rstrip().split(" ")[1])
        for line in file:
            tokens = line.rstrip().split(" ")
            embed_dict[tokens[0]] = torch.Tensor([float(weight) for weight in tokens[1:]])

    logging.info('Loaded {} / {} word embeddings'.format(
        len(set(embed_dict.keys()) & set(dictionary.words)), len(embed_dict)))
    embedding = nn.Embedding(len(dictionary), embed_dim, dictionary.pad_idx)
    for idx, word in enumerate(dictionary.words):
        if word in embed_dict:
            embedding.weight.data[idx] = embed_dict[word]
    return embedding


def move_to_cuda(sample):
    if torch.is_tensor(sample):
        return sample.cuda()
    elif isinstance(sample, list):
        return [move_to_cuda(x) for x in sample]
    elif isinstance(sample, dict):
        return {key: move_to_cuda(value) for key, value in sample.items()}
    else:
        return sample


def save_checkpoint(args, model, optimizer, epoch, valid_loss):
    os.makedirs(args.save_dir, exist_ok=True)
    last_epoch = getattr(save_checkpoint, 'last_epoch', -1)
    save_checkpoint.last_epoch = max(last_epoch, epoch)
    prev_best = getattr(save_checkpoint, 'best_loss', float('inf'))
    save_checkpoint.best_loss = min(prev_best, valid_loss)

    state_dict = {
        'epoch': epoch,
        'val_loss': valid_loss,
        'best_loss': save_checkpoint.best_loss,
        'last_epoch': save_checkpoint.last_epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'args': args,
    }

    if args.epoch_checkpoints and epoch % args.save_interval == 0:
        torch.save(state_dict, os.path.join(args.save_dir, 'checkpoint{}_{:.3f}.pt'.format(epoch, valid_loss)))
    if valid_loss < prev_best:
        torch.save(state_dict, os.path.join(args.save_dir, 'checkpoint_best.pt'))
    if last_epoch < epoch:
        torch.save(state_dict, os.path.join(args.save_dir, 'checkpoint_last.pt'))


def load_checkpoint(args, model, optimizer):
    checkpoint_path = os.path.join(args.save_dir, args.restore_file)
    if os.path.isfile(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=lambda s, l: default_restore_location(s, 'cpu'))
        model.load_state_dict(state_dict['model'])
        optimizer.load_state_dict(state_dict['optimizer'])
        save_checkpoint.best_loss = state_dict['best_loss']
        save_checkpoint.last_epoch = state_dict['last_epoch']
        logging.info('Loaded checkpoint {}'.format(checkpoint_path))
        return state_dict


def init_logging(args):
    handlers = [logging.StreamHandler()]
    if hasattr(args, 'log_file') and args.log_file is not None:
        os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
        handlers.append(logging.FileHandler(args.log_file, mode='w'))
    logging.basicConfig(handlers=handlers, format='[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)
    logging.info('COMMAND: %s' % ' '.join(sys.argv))
    logging.info('Arguments: {}'.format(vars(args)))


INCREMENTAL_STATE_INSTANCE_ID = defaultdict(lambda: 0)


def _get_full_incremental_state_key(module_instance, key):
    module_name = module_instance.__class__.__name__
    # Assign a unique ID to each module instance, so that incremental state is not shared across module instances
    if not hasattr(module_instance, '_fairseq_instance_id'):
        INCREMENTAL_STATE_INSTANCE_ID[module_name] += 1
        module_instance._fairseq_instance_id = INCREMENTAL_STATE_INSTANCE_ID[module_name]
    return '{}.{}.{}'.format(module_name, module_instance._fairseq_instance_id, key)


def get_incremental_state(module, incremental_state, key):
    """Helper for getting incremental state for an nn.Module."""
    full_key = _get_full_incremental_state_key(module, key)
    if incremental_state is None or full_key not in incremental_state:
        return None
    return incremental_state[full_key]


def set_incremental_state(module, incremental_state, key, value):
    """Helper for setting incremental state for an nn.Module."""
    if incremental_state is not None:
        full_key = _get_full_incremental_state_key(module, key)
        incremental_state[full_key] = value


def post_process_prediction(hypo_tokens, src_str, alignment, tgt_dict, remove_bpe):
    hypo_str = tgt_dict.string(hypo_tokens, remove_bpe)
    # hypo_str = replace_unk(hypo_str, src_str, alignment, tgt_dict.unk_word)
    # Convert back to tokens for evaluating with unk replacement or without BPE
    # Note that the dictionary can be modified inside the method.
    hypo_tokens = tgt_dict.binarize(hypo_str, preprocess.word_tokenize, add_if_not_exist=True)
    return hypo_tokens, hypo_str, alignment


def replace_unk(hypo_str, src_str, alignment, unk):
    hypo_tokens = preprocess.word_tokenize(hypo_str)
    src_tokens = preprocess.word_tokenize(src_str) + ['<eos>']
    for i, ht in enumerate(hypo_tokens):
        if ht == unk:
            hypo_tokens[i] = src_tokens[alignment[i]]
    return ' '.join(hypo_tokens)


def strip_pad(tensor, pad):
    return tensor[tensor.ne(pad)]

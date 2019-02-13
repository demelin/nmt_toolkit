import os
import argparse
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

import torch
from torch.serialization import default_restore_location

from seq2seq import models, utils
from seq2seq.data.dictionary import Dictionary
from seq2seq.data.dataset import Seq2SeqDataset, BatchSampler


def get_args():
    """ Defines training-specific hyper-parameters. """
    parser = argparse.ArgumentParser('Sequence to Sequence Model')
    parser.add_argument('--cuda', default=False, help='Use a GPU')

    # Add data arguments
    parser.add_argument('--data', default='prepared_data', help='path to data directory')
    parser.add_argument('--source-lang', default='jp', help='source language')
    parser.add_argument('--target-lang', default='en', help='target language')
    parser.add_argument('--checkpoint-path', default='checkpoints/checkpoint_best.pt',
                        help='path to the model file')
    parser.add_argument('--vis-dir', default='visualizations', help='path to the model file')
    return parser.parse_args()


def main(args):
    """ Main function. Visualizes attention weight arrays as nifty heat-maps. """
    mpl.rc('font', family='VL Gothic')

    torch.manual_seed(42)
    state_dict = torch.load(args.checkpoint_path, map_location=lambda s, l: default_restore_location(s, 'cpu'))
    args = argparse.Namespace(**{**vars(args), **vars(state_dict['args'])})
    utils.init_logging(args)

    # Load dictionaries
    src_dict = Dictionary.load(os.path.join(args.data, 'dict.{:s}'.format(args.source_lang)))
    print('Loaded a source dictionary ({:s}) with {:d} words'.format(args.source_lang, len(src_dict)))
    tgt_dict = Dictionary.load(os.path.join(args.data, 'dict.{:s}'.format(args.target_lang)))
    print('Loaded a target dictionary ({:s}) with {:d} words'.format(args.target_lang, len(tgt_dict)))

    # Load dataset
    test_dataset = Seq2SeqDataset(
        src_file=os.path.join(args.data, 'test.{:s}'.format(args.source_lang)),
        tgt_file=os.path.join(args.data, 'test.{:s}'.format(args.target_lang)),
        src_dict=src_dict, tgt_dict=tgt_dict)

    vis_loader = torch.utils.data.DataLoader(test_dataset, num_workers=1, collate_fn=test_dataset.collater,
                                             batch_sampler=BatchSampler(test_dataset, None, 1, 1, 0, shuffle=False,
                                                                        seed=42))

    # Build model and optimization criterion
    model = models.build_model(args, src_dict, tgt_dict)
    if args.cuda:
        model = model.cuda()
    model.load_state_dict(state_dict['model'])
    print('Loaded a model from checkpoint {:s}'.format(args.checkpoint_path))

    # Store attention weight arrays
    attn_records = list()

    # Iterate over the visualization set
    for i, sample in enumerate(vis_loader):
        if args.cuda:
            sample = utils.move_to_cuda(sample)
        if len(sample) == 0:
            continue

        # Perform forward pass
        output, attn_weights = model(sample['src_tokens'], sample['src_lengths'], sample['tgt_inputs'])
        attn_records.append((sample, attn_weights))

        # Only visualize the first 10 sentence pairs
        if i >= 10:
            break

    # Generate heat-maps and store them at the designated location
    if not os.path.exists(args.vis_dir):
        os.makedirs(args.vis_dir)

    for record_id, record in enumerate(attn_records):
        # Unpack
        sample, attn_map = record
        src_ids = utils.strip_pad(sample['src_tokens'].data, tgt_dict.pad_idx)
        tgt_ids = utils.strip_pad(sample['tgt_inputs'].data, tgt_dict.pad_idx)
        # Convert indices into word tokens
        src_str = src_dict.string(src_ids).split(' ') + ['<EOS>']
        tgt_str = tgt_dict.string(tgt_ids).split(' ') + ['<EOS>']

        # Generate heat-maps
        attn_map = attn_map.squeeze(dim=0).transpose(1, 0).detach().numpy()

        attn_df = pd.DataFrame(attn_map,
                               index=src_str,
                               columns=tgt_str)

        sns.heatmap(attn_df, cmap='Blues', linewidths=0.25, vmin=0.0, vmax=1.0, xticklabels=True, yticklabels=True,
                    fmt='.3f')
        plt.yticks(rotation=0)
        plot_path = os.path.join(args.vis_dir, 'sentence_{:d}.png'.format(record_id))
        plt.savefig(plot_path, dpi='figure', pad_inches=1, bbox_inches='tight')
        plt.clf()

    print('Done! Visualized attention maps have been saved to the \'{:s}\' directory!'.format(args.vis_dir))


if __name__ == '__main__':
    args = get_args()
    main(args)

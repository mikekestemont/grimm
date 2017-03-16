
import os
import sys
import time
import math

seed = 1001
import random
random.seed(seed)

import torch
try:
    torch.cuda.manual_seed(seed)
except:
    print('no NVIDIA driver found')
torch.manual_seed(seed)

import torch.nn as nn

from modules import ForkableLM, MultiheadLM, LMContainer
from optimizer import Optimizer
from dataset import Dict, CyclicBlockDataset
from early_stopping import EarlyStoppingException
import utils as u


def train_model_fork(
        model, train, valid, test, optim, epochs, criterion, d, **kwargs):
    print("Training main")
    test_ppl = u.train_model(
        model, train, valid, test, optim, epochs, criterion, d, **kwargs)
    model_J, model_W = model.fork_model(), model.fork_model()
    for label, model in [('J', model_J), ('W', model_W)]:
        print("Training brother [%s]" % label)
        optim.set_params(model.parameters())
        u.train_model(
            model, train, valid, test, optim, epochs, criterion, d, **kwargs)
    return {'J': model_J, 'W': model_W}, test_ppl


def train_model_multihead(
        model, train, valid, test, optim, epochs, criterion, d, **kwargs):
    print("Training main")
    test_ppl = u.train_model(
        model, train, valid, test, optim, epochs, criterion, d, **kwargs)
    return model, test_ppl


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', required=True)
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--dict_path', type=str)
    parser.add_argument('--already_split', action='store_true')
    parser.add_argument('--freeze_rnn', action='store_true')
    parser.add_argument('--freeze_emb', action='store_true')
    parser.add_argument('--training_mode', required=True)
    parser.add_argument('--max_size', default=1000000, type=int)
    parser.add_argument('--min_freq', default=1, type=int)
    parser.add_argument('--layers', default=2, type=int)
    parser.add_argument('--cell', default='GRU')
    parser.add_argument('--emb_dim', default=16, type=int)
    parser.add_argument('--hid_dim', default=248, type=int)
    parser.add_argument('--dropout', default=0.3, type=float)
    parser.add_argument('--tie_weights', action='store_true')
    parser.add_argument('--project_on_tied_weights', action='store_true')
    parser.add_argument('--batch_size', default=200, type=int)
    parser.add_argument('--bptt', default=10, type=int)
    parser.add_argument('--reset_hidden', action='store_true')
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--checkpoint', default=50, type=int)
    parser.add_argument('--hook', default=10, type=int,
                        help='Compute valid ppl after so many checkpoints')
    parser.add_argument('--early_stop', default=10, type=int)
    parser.add_argument('--optim', default='RMSprop', type=str)
    parser.add_argument('--learning_rate', default=0.01, type=float)
    parser.add_argument('--learning_rate_decay', default=0.5, type=float)
    parser.add_argument('--start_decay_at', default=5, type=int)
    parser.add_argument('--max_grad_norm', default=5., type=float)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--prefix', default='model', type=str)
    args = parser.parse_args()

    print('Loading data...')

    sys.path.append('../')
    from src.utils import filter_letters, load_letters, split, letters2lines

    def load_files(**kwargs):
        bpath = os.path.expanduser(args.path)
        letters = load_letters(bpath=bpath, **kwargs)
        return split(filter_letters(letters, min_len=0))

    if args.pretrained:
        assert args.dict_path, "Needs dict path for loading pretrained models"
        d = u.load_model(args.dict_path)
    else:
        d = Dict(max_size=args.max_size, min_freq=args.min_freq,
                 bos_token=u.BOS, eos_token=u.EOS)

    if args.already_split:      # fetch already splitted datasets
        train_J, train_W = load_files(subset='train/', start_from_line=0)
        train_J, train_W = letters2lines(train_J), letters2lines(train_W)
        test_J, test_W = load_files(subset='test/', start_from_line=0)
        test_J, test_W = letters2lines(test_J), letters2lines(test_W)
        if not d.fitted:
            d.fit(train_J, train_W)
        test = CyclicBlockDataset(
            {'J': test_J, 'W': test_W}, d, args.batch_size, args.bptt,
            gpu=args.gpu, evaluation=True)
        train, valid = CyclicBlockDataset(
            {'J': train_J, 'W': train_W}, d, args.batch_size, args.bptt,
            gpu=args.gpu).splits(test=0.1, dev=None)
    else:                       # fetch raw datasets computing splits
        J, W = load_files()
        J, W = letters2lines(J), letters2lines(W)
        if not d.fitted:
            d.fit(J, W)
        train, test, valid = CyclicBlockDataset(
            {'J': J, 'W': W}, d, args.batch_size, args.bptt,
            gpu=args.gpu).splits(dev=0.1)

    print(' * vocabulary size. %d' % len(d))
    print(" * number of train batches. %d" % (len(train) * args.bptt))
    print(" * number of test batches. %d" % (len(test) * args.bptt))
    print(" * number of valid batches. %d" % (len(valid) * args.bptt))

    print('Building model...')

    opts = {"num_layers": args.layers, "dropout": args.dropout,
            "tie_weights": args.tie_weights,
            "project_on_tied_weights": args.project_on_tied_weights}

    if args.training_mode.startswith('fork'):
        model_type, train_fn = ForkableLM, train_model_fork
    elif args.training_mode.startswith('multi'):
        model_type, train_fn = MultiheadLM, train_model_multihead
        opts.update({'heads': ('W', 'J')})
    else:
        raise ValueError("Unknown training mode [%s]" % args.training_mode)

    if args.pretrained:
        assert args.training_mode.startswith('multi'), \
            "pretrained only implemented for MultiheadLM"
        assert args.model_path, "Needs model path for loading pretrained model"
        pretrained = u.load_model(args.model_path)
        model = MultiheadLM.from_pretrained_model(pretrained, opts['heads'])
    else:
        model = model_type(
            len(d), args.emb_dim, args.hid_dim, cell=args.cell, **opts)

    model.apply(u.make_initializer())

    if args.freeze_rnn:
        model.freeze_submodule('rnn')
    if args.freeze_emb:
        model.freeze_submodule('embeddings')

    print(model)
    print(" * number of model parameters. %d" % model.n_params())

    optim = Optimizer(
        model.parameters(), args.optim,
        lr=args.learning_rate, max_norm=args.max_grad_norm,
        lr_decay=args.learning_rate_decay, start_decay_at=args.start_decay_at)
    criterion = nn.CrossEntropyLoss()

    try:
        start = time.time()
        trained_models, test_ppl = train_fn(
            model, train, valid, test, optim, args.epochs, criterion, d,
            gpu=args.gpu, early_stop=args.early_stop, hook=args.hook,
            checkpoint=args.checkpoint, reset_hidden=args.reset_hidden)
        if args.save:
            parent = '.'.join(os.path.basename(args.model_path).split('.')[:-1])
            if args.pretrained:
                f = '{prefix}.{parent}'.format(
                    prefix=args.prefix, parent=parent)
            else:
                f = '{prefix}.{cell}.{layers}l.{hid_dim}' + \
                    'h.{emb_dim}e.{bptt}b.{ppl}'
            fname = f.format(ppl="%.2f" % test_ppl, **vars(args))
            print("Saving model to [%s]..." % fname)
            lm = LMContainer(trained_models, d).to_disk(fname)
    except (EarlyStoppingException, KeyboardInterrupt) as e:
        print("Trained for [%d] secs" % (time.time() - start))
        test_ppl = math.exp(u.validate_model(model, test, criterion))
        print("Test perplexity: %g" % test_ppl)

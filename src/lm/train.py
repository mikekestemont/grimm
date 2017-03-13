
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
from dataset import Dict, BlockDataset
from early_stopping import EarlyStoppingException
import utils as u


def letters2lines(letters):
    return [list(line) for letter in letters for line in letter.lines]


def train_model_fork(
        model, train, valid, test, optim, epochs, criterion,
        gpu=False, early_stop=3, checkpoint=50, hook=10):
    print("Training main")
    test_ppl = u.train_model(
        model, train, valid, test, optim, epochs, criterion,
        gpu=gpu, checkpoint=checkpoint, hook=hook, early_stop=early_stop)
    model_J, model_W = model.fork_model(), model.fork_model()
    for label, model in [('J', model_J), ('W', model_W)]:
        print("Training brother [%s]" % label)
        optim.set_params(model.parameters())
        u.train_model(
            model, train, valid, test, optim, epochs, criterion,
            gpu=gpu, early_stop=early_stop, checkpoint=checkpoint,
            hook=hook, head=label)
    return {'J': model_J, 'W': model_W}, test_ppl


def train_model_multihead(
        model, train, valid, test, optim, epochs, criterion,
        gpu=False, early_stop=3, checkpoint=50, hook=10):
    print("Training main")
    test_ppl = u.train_model(model, train, valid, test, optim, epochs, criterion,
                  gpu=gpu, early_stop=early_stop, checkpoint=checkpoint,
                  hook=hook)
    return model, test_ppl


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', required=True)
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

    def load_files():
        sys.path.append('../')
        from src.utils import filter_letters, load_letters, split
        letters = load_letters(bpath=os.path.expanduser(args.path))
        return split(filter_letters(letters, min_len=0))

    J, W = load_files()
    J, W = letters2lines(J), letters2lines(W)
    d = Dict(max_size=args.max_size, min_freq=args.min_freq,
             bos_token=u.BOS, eos_token=u.EOS)
    d.fit(J, W)
    lines = {'J': [c for l in J for c in l], 'W': [c for l in W for c in l]}
    data = BlockDataset(lines, d, args.batch_size, args.bptt, gpu=args.gpu)
    train, test, valid = data.splits(dev=0.1)

    print(' * vocabulary size. %d' % len(d))

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

    model = model_type(len(d), args.emb_dim, args.hid_dim, cell=args.cell, **opts)

    model.apply(u.Initializer.make_initializer())
    print(model)

    optim = Optimizer(
        model.parameters(), args.optim,
        lr=args.learning_rate, max_norm=args.max_grad_norm,
        lr_decay=args.learning_rate_decay, start_decay_at=args.start_decay_at)
    criterion = nn.CrossEntropyLoss()

    start = time.time()

    try:
        trained_models, test_ppl = train_fn(
            model, train, valid, test, optim, args.epochs, criterion, gpu=args.gpu,
            early_stop=args.early_stop, checkpoint=args.checkpoint, hook=args.hook)
        if args.save:
            f = '{prefix}.{cell}.{layers}l.{hid_dim}h.{emb_dim}.{ppl}'
            fname = f.format(ppl=test_ppl, **vars(args))
            print("Saving model...")
            lm = LMContainer(trained_models, d).to_disk(fname)
    except (EarlyStoppingException, KeyboardInterrupt) as e:
        test_ppl = math.exp(u.validate_model(model, test, criterion))
        print("Test perplexity: %g" % test_ppl)
    finally:
        print("Trained for [%d] secs" % (time.time() - start))
        test_ppl = math.exp(u.validate_model(model, test, criterion))


import os
import sys
import time

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

from modules import ForkableLM, MultiheadLM
from optimizer import Optimizer
from dataset import Dict, BlockDataset
import utils as u


def letters2lines(letters):
    return [list(line) for letter in letters for line in letter.lines]


def train_model_fork(
        model, train, valid, test, optim, epochs, criterion,
        gpu=False, early_stop=3, checkpoint=50, hook=10):
    print("Training main")
    u.train_model(
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
    return {'J': model_J, 'W': model_W}


def train_model_multihead(
        model, train, valid, test, optim, epochs, criterion,
        gpu=False, early_stop=3, checkpoint=50, hook=10):
    print("Training main")
    u.train_model(model, train, valid, test, optim, epochs, criterion,
                  gpu=gpu, early_stop=early_stop, checkpoint=checkpoint,
                  hook=hook)
    return model


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
    train, test, valid = data.splits()

    print(' * vocabulary size. %d' % len(d))

    print('Building model...')

    if args.training_mode == 'fork':
        model_type, train_fn = ForkableLM, train_model_fork
    elif args.training_mode == 'multihead':
        model_type, train_fn = MultiheadLM, train_model_multihead
    else:
        raise ValueError("Unknown training mode [%s]" % args.training_mode)

    model = model_type(
        len(d), args.emb_dim, args.hid_dim, cell=args.cell,
        num_layers=args.layers, dropout=args.dropout,
        tie_weights=args.tie_weights, heads=('W', 'J'),
        project_on_tied_weights=args.project_on_tied_weights)

    model.apply(u.Initializer.make_initializer())
    print(model)

    optim = Optimizer(
        model.parameters(), args.optim, args.learning_rate, args.max_grad_norm,
        lr_decay=args.learning_rate_decay, start_decay_at=args.start_decay_at)
    criterion = nn.CrossEntropyLoss()

    start = time.time()

    training_out = train_fn(
        model, train, valid, test, optim, args.epochs, criterion,
        gpu=args.gpu, checkpoint=args.checkpoint, hook=args.hook)

    print("Trained for [%f] secs" % (time.time() - start))
    if args.save:
        f = '{prefix}.{cell}.{layers}l.{hid_dim}h.{emb_dim}.{training_mode}.pt'
        fname = f.format(**vars(args))
        if os.path.isfile(fname):
            answer = input(
                "File [%s] exists. Overwrite? (y/n): " % fname)
            if answer.lower() not in ("y", "yes"):
                print("Goodbye!")
                sys.exit(0)
        print("Saving model...")
        with open(fname, 'wb') as model_f, open(fname + 'dict') as dict_f:
            torch.save(training_out, model_f), torch.save(d, dict_f)

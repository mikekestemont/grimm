
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

from modules import LM, LMContainer
from optimizer import Optimizer
from dataset import Dict, BlockDataset
from early_stopping import EarlyStoppingException
import utils as u


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', required=True)
    parser.add_argument('--max_size', default=1000000, type=int)
    parser.add_argument('--min_freq', default=1, type=int)
    parser.add_argument('--layers', default=2, type=int)
    parser.add_argument('--cell', default='GRU')
    parser.add_argument('--emb_dim', default=16, type=int)
    parser.add_argument('--hid_dim', default=248, type=int)
    parser.add_argument('--dropout', default=0.3, type=float)
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

    sys.path.append('../')
    from src.utils import load_letters, letters2lines

    print("Loading data...")
    letters = load_letters(
        bpath=os.path.expanduser(args.path), subset='', start_from_line=0)
    lines = letters2lines(letters)
    d = Dict(max_size=args.max_size, min_freq=args.min_freq,
             bos_token=u.BOS, eos_token=u.EOS)
    d.fit(lines)
    train, test, valid = BlockDataset(
        lines, d, args.batch_size, args.bptt, gpu=args.gpu
    ).splits(test=0.1, dev=0.1)

    print(' * vocabulary size. %d' % len(d))

    print(" * number of train batches. %d" % (len(train) * args.bptt))
    print(" * number of test batches. %d" % (len(test) * args.bptt))
    print(" * number of valid batches. %d" % (len(valid) * args.bptt))

    print("Building model...")
    model = LM(len(d), args.emb_dim, args.hid_dim, cell=args.cell,
               num_layers=args.layers, dropout=args.dropout)

    model.apply(u.make_initializer())

    print(model)
    print(" * number of model parameters. %d" % model.n_params())

    optim = Optimizer(
        model.parameters(), args.optim,
        lr=args.learning_rate, max_norm=args.max_grad_norm,
        lr_decay=args.learning_rate_decay, start_decay_at=args.start_decay_at)
    criterion = nn.CrossEntropyLoss()

    try:
        start = time.time()
        trained_models, test_ppl = u.train_model(
            model, train, valid, test, optim, args.epochs, criterion,
            gpu=args.gpu, early_stop=args.early_stop, hook=args.hook,
            checkpoint=args.checkpoint, reset_hidden=args.reset_hidden)
        if args.save:
            f = '{prefix}.{cell}.{layers}l.{hid_dim}h.{emb_dim}e.{bptt}b.{ppl}'
            fname = f.format(ppl=int(test_ppl), **vars(args))
            print("Saving model to [%s]..." % fname)
            lm = LMContainer(trained_models, d).to_disk(fname)
    except (EarlyStoppingException, KeyboardInterrupt) as e:
        test_ppl = math.exp(u.validate_model(model, test, criterion))
        print("Test perplexity: %g" % test_ppl)
    finally:
        print("Trained for [%d] secs" % (time.time() - start))
        test_ppl = math.exp(u.validate_model(model, test, criterion))

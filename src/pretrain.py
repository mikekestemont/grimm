
import os

seed = 1001
import random                   # nopep8
random.seed(seed)

import torch                    # nopep8
try:
    torch.cuda.manual_seed(seed)
except:
    print('no NVIDIA driver found')
torch.manual_seed(seed)

import torch.nn as nn   # nopep8

from misc.optimizer import Optimizer                            # nopep8
from misc.dataset import Dict, BlockDataset, CyclicBlockDataset  # nopep8
from misc.trainer import LMTrainer                               # nopep8
from misc.loggers import StdLogger                               # nopep8
from misc.early_stopping import EarlyStopping                    # nopep8

from modules import utils as u      # nopep8
from modules.lm import LM           # nopep8

from container import LMContainer   # nopep8
from utils import load_letters, letters2lines, make_preprocessor  # nopep8


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
    parser.add_argument('--checkpoints_per_epoch', default=5, type=int)
    parser.add_argument('--early_stopping', default=10, type=int)
    parser.add_argument('--use_preprocessor', action='store_true')
    parser.add_argument('--optim', default='RMSprop', type=str)
    parser.add_argument('--learning_rate', default=0.01, type=float)
    parser.add_argument('--learning_rate_decay', default=0.5, type=float)
    parser.add_argument('--start_decay_at', default=5, type=int)
    parser.add_argument('--max_grad_norm', default=5., type=float)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--prefix', default='model', type=str)
    args = parser.parse_args()

    print("Loading data...")
    letters = load_letters(
        bpath=os.path.expanduser(args.path), subset='', start_from_line=0)
    preprocessor = make_preprocessor() if args.use_preprocessor else None
    lines = letters2lines(letters, preprocessor=preprocessor)
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
    if args.gpu:
        model.cuda()

    print(model)
    print(" * number of model parameters. %d" % model.n_params())

    optim = Optimizer(
        model.parameters(), args.optim, args.learning_rate, args.max_grad_norm,
        lr_decay=args.learning_rate_decay, start_decay_at=args.start_decay_at)
    criterion = nn.CrossEntropyLoss()

    datasets = {"train": train, "test": test, "valid": valid}
    trainer = LMTrainer(model, datasets, criterion, optim)

    if args.early_stopping > 0:
        early_stopping = EarlyStopping(args.early_stopping)
    model_check_hook = make_lm_check_hook(
        d, args.gpu, early_stopping=early_stopping)
    num_checks = len(train) // (args.checkpoint * args.checkpoints_per_epoch)
    trainer.add_hook(model_check_hook, num_checkpoints=num_checks)

    trainer.add_loggers(StdLogger())

    trainer.train(args.epochs, args.checkpoint, gpu=args.gpu)

    if args.save:
        test_ppl = trainer.validate_model(test=True)
        print("Test perplexity: %g" % test_ppl)
        if args.save:
            f = '{prefix}.{cell}.{layers}l.{hid_dim}h.{emb_dim}e.{bptt}b.{ppl}'
            fname = f.format(ppl="%.2f" % test_ppl, **vars(args))
            print("Saving model to [%s]..." % fname)
            u.save_model(model, fname, d=d)

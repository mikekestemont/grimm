
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
import torch.nn.functional as F
from torch.autograd import Variable

from modules import ForkableLM
from optimizer import Optimizer
from dataset import Dict
from preprocess import text_processor
import utils as u


def train_model_multihead(
        model, splits_J, splits_W, optim, epochs, bptt, criterion,
        gpu=False, early_stop=3, checkpoint=50, hook=10):
    model_J, model_W = model.fork_model(), model.fork_model()
    for head, submodel, (train, valid, test) in (
            [('Jacob', model_J, splits_J), ('Wilhelm', model_W, splits_W)]):
        print("Training head [%s]" % head)
        print(" * number of train batches. %d" % len(train))
        n_params = sum([p.nelement() for p in submodel.parameters()])
        print(" * number of parameters. %d" % n_params)
        optim.set_params(submodel.parameters())
        train_model(
            submodel, train, valid, test, optim, epochs, bptt, criterion,
            gpu=gpu, early_stop=early_stop, checkpoint=checkpoint, hook=hook)
    return model_J, model_W


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', required=True)
    parser.add_argument('--max_size', default=1000000, type=int)
    parser.add_argument('--min_freq', default=1, type=int)
    parser.add_argument('--layers', default=2, type=int)
    parser.add_argument('--cell', default='GRU')
    parser.add_argument('--emb_dim', default=200, type=int)
    parser.add_argument('--hid_dim', default=200, type=int)
    parser.add_argument('--dropout', default=0.3, type=float)
    parser.add_argument('--tie_weights', action='store_true')
    parser.add_argument('--project_on_tied_weights', action='store_true')
    parser.add_argument('--batch_size', default=200, type=int)
    parser.add_argument('--bptt', default=20, type=int)
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
        import sys, os
        sys.path.append('../')
        from src.utils import load_letters, split
        letters = load_letters(bpath=os.path.expanduser(args.path))
        return split(letters)

    def letters2chars(letters):
        random.shuffle(letters)
        return ' '.join([c for l in letters for w in l.words for c in w])

    def load_data(chars, d):
        return batchify(
            torch.LongTensor(list(d.transform(chars))),
            args.batch_size, gpu=args.gpu)

    (train_J, test_J, valid_J), (train_W, test_W, valid_W) = load_files()
    assert len(train_J) > 0, "Didn't find letters"
    train, test, valid = train_J + train_W, test_J + test_W, valid_J + valid_W
    train = letters2chars(train).split()
    test = letters2chars(test).split()
    valid = letters2chars(valid).split()
    d = Dict(max_size=args.max_size, min_freq=args.min_freq, sequential=False)
    d.fit(train, test, valid)
    train = load_data(train, d)
    test = load_data(test, d)
    valid = load_data(valid, d)
    train_J = letters2chars(train_J).split()
    train_W = letters2chars(train_W).split()
    test_J = letters2chars(test_J).split()
    test_W = letters2chars(test_W).split()
    valid_J = letters2chars(valid_J).split()
    valid_W = letters2chars(valid_W).split()
    train_J, train_W = load_data(train_J, d), load_data(train_W, d)
    test_J, test_W = load_data(test_J, d), load_data(test_W, d)
    valid_J, valid_W = load_data(valid_J, d), load_data(valid_W, d)

    print(' * vocabulary size. %d' % len(d))
    print(' * number of train batches. %d' % len(train))

    print('Building model...')
    model = ForkableLM(len(d), args.emb_dim, args.hid_dim,
               num_layers=args.layers, cell=args.cell,
               dropout=args.dropout, tie_weights=args.tie_weights,
               project_on_tied_weights=args.project_on_tied_weights)

    model.apply(u.Initializer.make_initializer())

    n_params = sum([p.nelement() for p in model.parameters()])
    print('* number of parameters: %d' % n_params)
    print(model)

    optim = Optimizer(
        model.parameters(), args.optim, args.learning_rate, args.max_grad_norm,
        lr_decay=args.learning_rate_decay, start_decay_at=args.start_decay_at)
    criterion = nn.CrossEntropyLoss()

    start = time.time()
    try:
        train_model(
            model, train, valid, test, optim, args.epochs, args.bptt,
            criterion, gpu=args.gpu, checkpoint=args.checkpoint,
            hook=args.hook)
        model_J, model_W = train_model_multihead(
            model, (train_J, valid_J, test_J), (train_W, valid_W, test_W),
            optim, args.epochs, args.bptt, criterion, gpu=args.gpu,
            checkpoint=args.checkpoint, hook=args.hook)
    except u.EarlyStopping as e:
        print(e.message)
        pass
    except KeyboardInterrupt:
        pass
    finally:
        print("Trained for [%f] secs" % (time.time() - start))
        if args.save:
            import os
            import sys
            test_loss = validate_model(
                model, test, args.bptt, criterion, args.gpu)
            test_ppl = math.exp(test_loss)
            f = '{prefix}.{cell}.{layers}l.{hid_dim}h.{emb_dim}e.{ppl}.pt'
            filename = f.format(ppl=int(test_ppl), **vars(args))
            if os.path.isfile(filename):
                answer = input(
                    "File [%s] exists. Overwrite? (y/n): " % filename)
                if answer.lower() not in ("y", "yes"):
                    print("Goodbye!")
                    sys.exit(0)
            print("Saving model...")
            with open(filename, 'wb') as f:
                torch.save(model, f)

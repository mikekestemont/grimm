
import os
import sys

seed = 1001
import random                   # nopep8
random.seed(seed)

import torch                    # nopep8
try:
    torch.cuda.manual_seed(seed)
except:
    print('no NVIDIA driver found')
torch.manual_seed(seed)

import torch.nn as nn           # nopep8

from modules import MultiheadLM, LMContainer  # nopep8
from optimizer import Optimizer               # nopep8
from dataset import Dict, CyclicBlockDataset  # nopep8
from trainer import LMTrainer, Logger         # nopep8
from early_stopping import EarlyStoppingException, EarlyStopping  # nopep8
import utils as u                                                 # nopep8


def make_lm_check_hook(d, gpu, early_stopping):

    def hook(trainer, batch_num, checkpoint):
        print("Checking training...")
        loss = trainer.validate_model()
        print("Valid loss: %g" % loss)
        print("Registering early stopping loss...")
        if early_stopping is not None:
            early_stopping.add_checkpoint(loss)
        print("Generating text...")
        print("***")
        if isinstance(trainer.datasets["train"], CyclicBlockDataset):
            for head in trainer.datasets["train"].names:
                scores, hyps = trainer.model.generate_beam(
                    d.get_bos(), d.get_eos(),
                    head=head, gpu=gpu, max_seq_len=100)
                print(' * [%s]' % head)
                u.print_hypotheses(scores, hyps, d)
        else:
            scores, hyps = trainer.model.generate_beam(
                d.get_bos(), d.get_eos(), gpu=gpu, max_seq_len=100)
            u.print_hypotheses(scores, hyps, d)
        print("***")

    return hook


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
    parser.add_argument('--use_preprocessor', action='store_true')
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--checkpoint', default=50, type=int)
    parser.add_argument('--checkpoints_per_epoch', default=5, type=int)
    parser.add_argument('--early_stopping', default=10, type=int)
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
    from src.utils import filter_letters, load_letters, \
        split, letters2lines, make_preprocessor

    preprocessor = make_preprocessor() if args.use_preprocessor else None

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
        train_J = letters2lines(train_J, preprocessor=preprocessor)
        train_W = letters2lines(train_W, preprocessor=preprocessor)
        test_J, test_W = load_files(subset='test/', start_from_line=0)
        test_J = letters2lines(test_J, preprocessor=preprocessor)
        test_W = letters2lines(test_W, preprocessor=preprocessor)
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
        J = letters2lines(J, preprocessor=preprocessor)
        W = letters2lines(W, preprocessor=preprocessor)
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

    heads = ('W', 'J')

    if args.pretrained:
        assert args.model_path, "Needs model path for loading pretrained model"
        pretrained = u.load_model(args.model_path)
        model = MultiheadLM.from_pretrained_model(pretrained, heads)
    else:
        model = MultiheadLM(
            len(d), args.emb_dim, args.hid_dim, cell=args.cell,
            num_layers=args.layers, dropout=args.dropout,
            tie_weights=args.tie_weights,
            project_on_tied_weights=args.project_on_tied_weights,
            heads=('W', 'J'))

    model.apply(u.make_initializer())

    if args.gpu:
        model.cuda()

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

    datasets = {'train': train, 'valid': valid, 'test': test}
    trainer = LMTrainer(model, datasets, criterion, optim)

    if args.early_stopping > 0:
        early_stopping = EarlyStopping(args.early_stopping)
    model_check_hook = make_lm_check_hook(
        d, args.gpu, early_stopping=early_stopping)
    num_checks = len(train) // (args.checkpoint * args.checkpoints_per_epoch)
    trainer.add_hook(model_check_hook, num_checkpoints=num_checks)

    trainer.add_loggers(Logger())

    trainer.train(args.epochs, args.checkpoint, gpu=args.gpu)

    if args.save:
        loss = trainer.validate_model(test=True)
        parent = '.'.join(os.path.basename(args.model_path).split('.')[:-1])
        if args.pretrained:
            f = '{prefix}.{parent}'.format(prefix=args.prefix, parent=parent)
        else:
            f = '{prefix}.{cell}.{layers}l.{hid_dim}h.{emb_dim}e.{bptt}b.{ppl}'
        fname = f.format(ppl="%.2f" % loss, **vars(args))
        print("Saving model to [%s]..." % fname)
        LMContainer(model, d).to_disk(fname)

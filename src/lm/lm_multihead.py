
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

from modules import StackedRNN, TiedEmbedding, TiedLinear
from optimizer import Optimizer
from dataset import Dict
from preprocess import text_processor
import utils as u


class LM(nn.Module):
    def __init__(self, vocab, emb_dim, hid_dim, num_layers=1,
                 cell='LSTM', bias=True, dropout=0.0, tie_weights=False,
                 project_on_tied_weights=False):
        self.vocab = vocab
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        if tie_weights and not project_on_tied_weights:
            assert self.emb_dim == self.hid_dim, \
                "When tying weights, output projection and " + \
                "embedding layer should have equal size"
        self.num_layers = num_layers
        self.cell = cell
        self.bias = bias
        self.has_dropout = bool(dropout)
        self.dropout = dropout

        super(LM, self).__init__()
        weight = None
        if tie_weights:
            weight = nn.parameter.Parameter(torch.randn(vocab, emb_dim))
            self.embeddings = TiedEmbedding(vocab, self.emb_dim, weight)
        else:
            self.embeddings = nn.Embedding(vocab, self.emb_dim)
        self.rnn = getattr(nn, cell)(
            self.emb_dim, self.hid_dim,
            num_layers=num_layers, bias=bias, dropout=dropout)
        if tie_weights:
            if self.emb_dim == self.hid_dim:
                self.project = TiedLinear(self.hid_dim, vocab, weight)
            else:
                assert project_on_tied_weights
                self.project = nn.Sequential(
                    nn.Linear(self.hid_dim, self.emb_dim),
                    TiedLinear(self.hid_dim, vocab, weight))
        else:
            self.project = nn.Linear(self.hid_dim, vocab)

    def parameters(self):
        for p in super(LM, self).parameters():
            if p.requires_grad is not False:
                yield p

    def freeze_submodule(self, module):
        for p in getattr(self, module).parameters():
            p.requires_grad = False

    def fork_model(self):
        model = LM(self.vocab, self.emb_dim, self.hid_dim,
                   num_layers=self.num_layers, cell=self.cell, bias=self.bias,
                   dropout=self.dropout, tie_weights=False,
                   project_on_tied_weights=False)
        model.load_dict(self.state_dict())
        model.freeze_submodule('embeddings')
        return model

    def init_hidden_for(self, inp):
        batch = inp.size(1)
        size = (self.num_layers, batch, self.hid_dim)
        h_0 = Variable(inp.data.new(*size).zero_(), requires_grad=False)
        if self.cell.startswith('GRU'):
            return h_0
        else:
            c_0 = Variable(inp.data.new(*size).zero_(), requires_grad=False)
            return h_0, c_0

    def forward(self, inp, hidden=None):
        emb = self.embeddings(inp)
        if self.has_dropout:
            emb = F.dropout(emb, p=self.dropout, training=self.training)
        outs, hidden = self.rnn(emb, hidden or self.init_hidden_for(emb))
        if self.has_dropout:
            outs = F.dropout(outs, p=self.dropout, training=self.training)
        seq_len, batch, hid_dim = outs.size()
        # (seq_len x batch x hid) -> (seq_len * batch x hid)
        logs = self.project(outs.view(seq_len * batch, hid_dim))
        return logs, hidden

    def generate_beam(self, bos, eos, max_seq_len=20, width=5, gpu=False):
        "Generate text using beam search decoding"
        beam = Beam(width, bos, eos, gpu=gpu)
        hidden = self.init_hidden_for(beam.get_current_state())
        while beam.active and len(beam) < max_seq_len:
            prev = Variable(
                beam.get_current_state().unsqueeze(0), volatile=True)
            logs, hidden = self(prev, hidden=hidden)
            beam.advance(logs.data)
            if self.cell.startswith('LSTM'):
                hidden = (u.swap(hidden[0], 1, beam.get_source_beam()),
                          u.swap(hidden[1], 1, beam.get_source_beam()))
            else:
                hidden = u.swap(hidden, 1, beam.get_source_beam())
        scores, hyps = beam.decode()
        return hyps

    def generate(self, bos, eos, max_seq_len=20, beam=None, gpu=False):
        "Generate text using simple argmax decoding"
        prev = Variable(torch.LongTensor([bos]).unsqueeze(0), volatile=True)
        if gpu: prev = prev.cuda()
        hidden, hyp = None, []
        for _ in range(max_seq_len):
            logs, hidden = self(prev, hidden=hidden)
            prev = logs.max(1)[1].t()
            hyp.append(prev)
            if prev.data.eq(eos).nonzero().nelement() > 0:
                break
        return [hyp]


# Load data
def batchify(data, batch_size, gpu=False):
    num_batches = len(data) // batch_size
    data = data.narrow(0, 0, num_batches * batch_size)
    data = data.view(batch_size, -1).t().contiguous()
    if gpu:
        data = data.cuda()
    return data


def get_batch(data, i, bptt, evaluation=False, gpu=False):
    seq_len = min(bptt, len(data) - 1 - i)
    src = Variable(data[i:i+seq_len], volatile=evaluation)
    trg = Variable(data[i+1:i+seq_len+1].view(-1), volatile=evaluation)
    if gpu:
        src, trg = src.cuda(), trg.cuda()
    return src, trg


# Training code
def make_criterion(vocab_size, mask_ids=()):
    weight = torch.ones(vocab_size)
    for mask in mask_ids:
        weight[mask] = 0
    return nn.CrossEntropyLoss(weight=weight)


def repackage_hidden(h):
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


def validate_model(model, data, bptt, criterion, gpu):
    loss, hidden = 0, None
    for i in range(0, len(data) - 1, bptt):
        source, targets = get_batch(data, i, bptt, evaluation=True, gpu=gpu)
        output, hidden = model(source, hidden=hidden)
        # since loss is averaged across observations for each minibatch
        loss += len(source) * criterion(output, targets).data[0]
        hidden = repackage_hidden(hidden)
    return loss / len(data)


def train_epoch(model, data, optim, criterion, bptt, epoch, checkpoint, gpu,
                hook=0, on_hook=None):
    """
    hook: compute `on_hook` every `hook` checkpoints
    """
    epoch_loss, batch_loss, report_words = 0, 0, 0
    start = time.time()
    hidden = None

    for batch, i in enumerate(range(0, len(data) - 1, bptt)):
        model.zero_grad()
        source, targets = get_batch(data, i, bptt, gpu=gpu)
        output, hidden = model(source, hidden)
        loss = criterion(output, targets)
        hidden = repackage_hidden(hidden)
        loss.backward(), optim.step()
        # since loss is averaged across observations for each minibatch
        epoch_loss += len(source) * loss.data[0]
        batch_loss += loss.data[0]
        report_words += targets.nelement()

        if batch % checkpoint == 0 and batch > 0:
            print("Epoch %d, %5d/%5d batches; ppl: %6.2f; %3.0f tokens/s" %
                  (epoch, batch, len(data) // bptt,
                   math.exp(batch_loss / checkpoint),
                   report_words / (time.time() - start)))
            report_words = batch_loss = 0
            start = time.time()
            # call thunk every `hook` checkpoints
            if hook and (batch // checkpoint) % hook == 0:
                if on_hook is not None:
                    on_hook(batch // checkpoint)
    return epoch_loss / len(data)


def train_model(model, train, valid, test, optim, epochs, bptt,
                criterion, gpu=False, early_stop=3, checkpoint=50, hook=10):
    if gpu:
        criterion.cuda(), model.cuda()

    # hook function
    last_val_ppl, num_idle_hooks = float('inf'), 0

    def on_hook(checkpoint):
        nonlocal last_val_ppl, num_idle_hooks
        model.eval()
        valid_loss = validate_model(model, valid, bptt, criterion, gpu)
        if optim.method == 'SGD':
            last_lr, new_lr = optim.maybe_update_lr(checkpoint, valid_loss)
            if last_lr != new_lr:
                print("Decaying lr [%f -> %f]" % (last_lr, new_lr))
        if valid_loss >= last_val_ppl:  # update idle checkpoints
            num_idle_hooks += 1
        last_val_ppl = valid_loss
        if num_idle_hooks >= early_stop:  # check for early stopping
            raise u.EarlyStopping(
                "Stopping after %d idle checkpoints" % num_idle_hooks, {})
        model.train()
        print("Valid perplexity: %g" % math.exp(min(valid_loss, 100)))

    for epoch in range(1, epochs + 1):
        # train
        model.train()
        train_loss = train_epoch(
            model, train, optim, criterion, bptt, epoch, checkpoint, gpu,
            hook=hook, on_hook=on_hook)
        print("Train perplexity: %g" % math.exp(min(train_loss, 100)))
        # val
        model.eval()
        valid_loss = validate_model(model, valid, bptt, criterion, gpu)
        print("Valid perplexity: %g" % math.exp(min(valid_loss, 100)))
    # test
    test_loss = validate_model(model, test, bptt, criterion, gpu)
    print("Test perplexity: %g" % math.exp(test_loss))
    return math.exp(test_loss)


def train_model_multihead(
        model, splits_J, splits_W, optim, epochs, bptt, criterion,
        gpu=False, early_stop=3, checkpoint=50, hook=10):
    model_J, model_W = model.fork_model(), model.fork_model()
    for idx, (submodel, (train, valid, test)) in enumerate(
            [(model_J, splits_J), (model_W, splits_W)]):
        print("Training head [%d]" % idx)
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
    parser.add_argument('--cell', default='LSTM')
    parser.add_argument('--emb_dim', default=200, type=int)
    parser.add_argument('--hid_dim', default=200, type=int)
    parser.add_argument('--dropout', default=0.3, type=float)
    parser.add_argument('--tie_weights', action='store_true')
    parser.add_argument('--project_on_tied_weights', action='store_true')
    parser.add_argument('--batch_size', default=20, type=int)
    parser.add_argument('--bptt', default=20, type=int)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--checkpoint', default=200, type=int)
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
        import sys
        sys.path.append('../')
        from src.utils import load_letters, split
        letters = load_letters(bpath=args.path)
        return split(letters)

    def letters2chars(letters):
        random.shuffle(letters)
        return ' '.join([c for l in letters for w in l.words for c in w])

    def load_data(chars, d):
        return batchify(
            torch.LongTensor(list(d.transform(chars))),
            args.batch_size, gpu=args.gpu)

    (train_J, test_J, valid_J), (train_W, test_W, valid_W) = load_files()
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
    model = LM(len(d), args.emb_dim, args.hid_dim,
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
        train_model_multihead(
            model, (train_J, valid_J, test_J), (train_W, valid_W, test_W),
            optim, args.epochs, args.bptt, criterion, gpu=args.gpu,
            checkpoint=args.checkpoint, hook=args.hook)
    except u.EarlyStopping:
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

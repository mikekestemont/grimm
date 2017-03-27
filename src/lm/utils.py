
import time
import math
from collections import OrderedDict
from itertools import groupby

import torch
import torch.nn.init as init
from torch.autograd import Variable

from dataset import CyclicBlockDataset
from early_stopping import EarlyStopping

from torch import nn


BOS = '<bos>'
EOS = '<eos>'
PAD = '<pad>'


# General utils
def load_model(path):
    if path.endswith('pickle'):
        import pickle as p
        load_fn = p.load
    elif path.endswith('pt'):
        import torch
        load_fn = torch.load
    else:
        raise ValueError("Unknown file format [%s]" % path)
    with open(path, 'rb') as f:
        return load_fn(f)


def save_model(model, prefix, d=None, mode='torch'):
    if mode == 'torch':
        import torch
        save_fn, ext = torch.save, 'pt'
    elif mode == 'pickle':
        import pickle as p
        save_fn, ext = p.dump, 'pickle'
    else:
        raise ValueError("Unknown mode [%s]" % mode)
    with open(prefix + "." + ext, 'wb') as f:
        save_fn(model, f)
    if d is not None:
        with open(prefix + ".dict." + ext, 'wb') as f:
            save_fn(d, f)


# Pytorch utils
def merge_states(state_dict1, state_dict2, merge_map):
    """
    Merge 2 state_dicts mapping parameters in state_dict2 to parameters
    in state_dict1 according to a dict mapping parameter names in
    state_dict2 to parameter names in state_dict1.
    """
    state_dict = OrderedDict()
    for p in state_dict2:
        if p in merge_map:
            target_p = merge_map[p]
            assert target_p in state_dict1, \
                "Wrong target param [%s]" % target_p
            state_dict[target_p] = [p]
        else:
            state_dict[target_p] = state_dict1[target_p]
    return state_dict


def tile(t, times):
    """
    Repeat a tensor across an added first dimension a number of times
    """
    return t.unsqueeze(0).expand(times, *t.size())


def swap(x, dim, perm):
    """
    swap the entries of a tensor in given dimension according to a given perm
    - dim: int,
        It has to be less than total number of dimensions in x.
    - perm: list or torch.LongTensor,
        It has to be as long as the specified dimension and it can only contain
        unique elements.
    """
    if isinstance(perm, list):
        perm = torch.LongTensor(perm)
    return Variable(x.data.index_select(dim, perm))


def repackage_bidi(h_or_c):
    """
    In a bidirectional RNN output is (output, (h_n, c_n))
    - output: (seq_len x batch x hid_dim * 2)
    - h_n: (num_layers * 2 x batch x hid_dim)
    - c_n: (num_layers * 2 x batch x hid_dim)

    This function turns a hidden input into:
        (num_layers x batch x hid_dim * 2)
    """
    layers_2, bs, hid_dim = h_or_c.size()
    return h_or_c.view(layers_2 // 2, 2, bs, hid_dim) \
                 .transpose(1, 2).contiguous() \
                 .view(layers_2 // 2, bs, hid_dim * 2)


def unpackage_bidi(h_or_c):
    layers, bs, hid_dim_2 = h_or_c.size()
    return h_or_c.view(layers, bs, 2, hid_dim_2 // 2) \
                 .transpose(1, 2).contiguous() \
                 .view(layers * 2, bs, hid_dim_2 // 2)


def map_index(t, source_idx, target_idx):
    """
    Map a source integer to a target integer across all dims of a LongTensor.
    """
    return t.masked_fill_(t.eq(source_idx), target_idx)


def select_cols(t, vec):
    """
    Select columns in t according to vec.

    Parameters:
    -----------
    - t: torch.Tensor (m x n)
    - vec: list with indices of length m with the longest integer at most n.
    """
    nrows, ncols = t.size()
    rows = torch.LongTensor(list(range(ncols))).repeat(nrows, 1)
    vec = torch.LongTensor(vec).unsqueeze(1).repeat(1, ncols)
    return t[rows == vec]


# Initializers
def default_weight_init(m, init_range=0.05):
    for p in m.parameters():
        p.data.uniform_(-init_range, init_range)


def rnn_param_type(param):
    """
    Distinguish between bias and state weight params inside an RNN.
    Criterion: biases are vectors, weights are matrices.
    """
    if len(param.size()) == 2:
        return "weight"
    elif len(param.size()) == 1:
        return "bias"
    raise ValueError("Unkown param shape of size [%d]" % param.size())


def make_initializer(
        linear={'type': 'uniform', 'args': {'a': -0.05, 'b': 0.05}},
        rnn={'type': 'xavier_uniform', 'args': {'gain': 1.}},
        rnn_bias={'type': 'constant', 'args': {'val': 0.}},
        emb={'type': 'uniform', 'args': {'a': -0.05, 'b': 0.05}},
        default={'type': 'uniform', 'args': {'a': -0.05, 'b': 0.05}}):
        """
        Creates an initializer function customizable on a layer basis.
        """
        rnns = (torch.nn.LSTM, torch.nn.GRU,
                torch.nn.LSTMCell, torch.nn.GRUCell)

        def initializer(m):
            if isinstance(m, (rnns)):  # RNNs
                for p_type, ps in groupby(m.parameters(), rnn_param_type):
                    if p_type == 'weight':
                        for p in ps:
                            getattr(init, rnn['type'])(p, **rnn['args'])
                    else:       # bias
                        for p in ps:
                            getattr(init, rnn_bias['type'])(
                                p, **rnn_bias['args'])
            elif isinstance(m, torch.nn.Linear):  # linear
                for param in m.parameters():
                    getattr(init, linear['type'])(param, **linear['args'])
            elif isinstance(m, torch.nn.Embedding):  # embedding
                for param in m.parameters():
                    getattr(init, emb['type'])(param, **emb['args'])
            else:               # default initializer
                for param in m.parameters():
                    getattr(init, 'uniform')(param, a=-0.05, b=0.05)
        return initializer


# Hooks
def has_trainable_parameters(module):
    """
    Determines whether a given module has any trainable parameters at all
    as per parameter.requires_grad
    """
    num_trainable_params = sum(p.requires_grad for p in module.parameters())
    return num_trainable_params > 0


def log_grad(module, grad_input, grad_output):
    """
    Logs input and output gradients for a given module.
    To be used by module.register_backward_hook or module.register_forward_hook
    """
    if not has_trainable_parameters(module):
        return

    def grad_norm_str(var):
        if isinstance(var, tuple) and len(var) > 1:
            return ", ".join("%6.4f" % g.data.norm() for g in var)
        else:
            if isinstance(var, tuple):
                return "%6.4f" % var[0].data.norm()
            return "%6.4f" % var.data.norm()

    log = "{module}: grad input [{grad_input}], grad output [{grad_output}]"
    print(log.format(
        module=module.__class__.__name__,
        grad_input=grad_norm_str(grad_input),
        grad_output=grad_norm_str(grad_output)))


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


def validate_model(model, data, criterion, subset=None, reset_hidden=False):
    loss, num_words, hidden = 0, 0, None
    for i in range(len(data)):
        if isinstance(data, CyclicBlockDataset):
            source, targets, head = data[i]
            if subset is not None and subset != head:
                continue
            output, hidden = model(source, hidden=hidden, head=head)
        else:
            source, targets = data[i]
            output, hidden = model(source, hidden=hidden)
            # since loss is averaged across observations for each minibatch
        loss += len(source) * criterion(output, targets).data[0]
        num_words += len(source)
        if reset_hidden:
            hidden.data.zero_()
        hidden = repackage_hidden(hidden)
    return loss / num_words


def train_epoch(model, data, optim, criterion, epoch, checkpoint,
                hook=0, on_hook=None, subset=None, reset_hidden=False):
    """
    hook: compute `on_hook` every `hook` checkpoints
    """
    epoch_loss, report_loss, report_words, epoch_words = 0, 0, 0, 0
    hidden = None
    start = time.time()

    for i in range(len(data)):
        model.zero_grad()
        if isinstance(data, CyclicBlockDataset):
            source, targets, head = data[i]
            if subset is not None and subset != head:
                continue
            output, hidden = model(source, hidden=hidden, head=head)
        else:
            source, targets = data[i]
            output, hidden = model(source, hidden=hidden)
        loss = criterion(output, targets)
        if reset_hidden:
            hidden.data.zero_()
        hidden = repackage_hidden(hidden)
        loss.backward(), optim.step()
        # since loss is averaged across observations for each minibatch
        epoch_loss += len(source) * loss.data[0]
        report_loss += len(source) * loss.data[0]
        report_words += len(source)
        epoch_words += len(source)

        if i % checkpoint == 0 and i > 0:
            print("Epoch %d, %5d/%5d batches; ppl: %6.2f; %3.0f tokens/s" %
                  (epoch, i, len(data), math.exp(report_loss / report_words),
                   report_words / (time.time() - start)))
            report_words = report_loss = 0
            start = time.time()
            # call thunk every `hook` checkpoints
            if hook and (i // checkpoint) % hook == 0:
                if on_hook is not None:
                    on_hook(i // checkpoint)
    return epoch_loss / epoch_words


def print_hypotheses(scores, hyps, d):
    for idx, (score, hyp) in enumerate(zip(scores, hyps)):
        print('*** Hypothesis %d' % (idx + 1))
        print('* ' + ''.join([d.vocab[c] for c in hyp]))
        print('* Sentence score: %g' % (score / len(hyp)))


def train_model(model, train, valid, test, optim, epochs, criterion, d,
                gpu=False, early_stop=5, checkpoint=50, hook=10, subset=None,
                reset_hidden=False, max_seq_len=100):
    if gpu:
        criterion.cuda(), model.cuda()

    early_stop = EarlyStopping(early_stop)

    def on_hook(checkpoint):
        model.eval()
        valid_ppl = math.exp(validate_model(
            model, valid, criterion, subset=subset, reset_hidden=reset_hidden))
        # log checkpoint
        print("Valid perplexity: %g" % valid_ppl)
        # generate a sentence
        print("Generating text...")
        if isinstance(train, CyclicBlockDataset):
            for head in train.names:
                scores, hyps = model.generate_beam(
                    d.get_bos(), d.get_eos(),
                    gpu=gpu, head=head, max_seq_len=max_seq_len)
                print('* [%s]: ' % head)
                print_hypotheses(scores, hyps, d)
        else:
            scores, hyps = model.generate_beam(
                d.get_bos(), d.get_eos(), gpu=gpu, max_seq_len=max_seq_len)
            print_hypotheses(scores, hyps, d)
        print("***")
        # maybe decay lr
        if optim.method == 'SGD':
            last_lr, new_lr = optim.maybe_update_lr(checkpoint, valid_ppl)
            if last_lr != new_lr:
                print("Decaying lr [%f -> %f]" % (last_lr, new_lr))
        # early stopping
        early_stop.add_checkpoint(valid_ppl)
        model.train()

    for epoch in range(1, epochs + 1):
        print("Starting epoch [%d]" % epoch)
        # train
        model.train()
        train_loss = train_epoch(
            model, train, optim, criterion, epoch, checkpoint,
            hook=hook, on_hook=on_hook, subset=subset)
        print("Train perplexity: %g" % math.exp(min(train_loss, 100)))
        # val
        model.eval()
        valid_loss = validate_model(
            model, valid, criterion, subset=subset, reset_hidden=reset_hidden)
        print("Valid perplexity: %g" % math.exp(min(valid_loss, 100)))
    # test
    test_loss = validate_model(
        model, test, criterion, subset=subset, reset_hidden=reset_hidden)
    print("Test perplexity: %g" % math.exp(test_loss))
    return math.exp(test_loss)

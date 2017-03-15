
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import utils as u
from beam_search import Beam


class StackedRNN(nn.Module):
    def __init__(self, num_layers, in_dim, hid_dim, cell='LSTM', dropout=0.0):
        self.num_layers = num_layers
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.cell = cell
        self.has_dropout = bool(dropout)
        self.dropout = dropout
        super(StackedRNN, self).__init__()

        for i in range(self.num_layers):
            layer = getattr(nn, cell + 'Cell')(in_dim, hid_dim)
            self.add_module(cell + 'Cell_%d' % i, layer)
            in_dim = hid_dim

    def forward(self, inp, hidden):
        """
        Parameters:
        ==========
        - inp: torch.Tensor (batch x inp_dim),
            Tensor holding the target for the previous decoding step,
            inp_dim = emb_dim or emb_dim + hid_dim if self.add_pred is True.
        - hidden: tuple (h_c, c_0), output of previous step
            h_c: (num_layers x batch x hid_dim)
            n_c: (num_layers x batch x hid_dim)

        Returns: output, (h_n, c_n)
        ==========
        - output: torch.Tensor (batch x hid_dim)
        - h_n: torch.Tensor (num_layers x batch x hid_dim)
        - c_n: torch.Tensor (num_layers x batch x hid_dim)
        """
        if self.cell.startswith('LSTM'):
            h_0, c_0 = hidden
            h_1, c_1 = [], []
            for i in range(self.num_layers):
                layer = getattr(self, self.cell + ('Cell_%d' % i))
                h_1_i, c_1_i = layer(inp, (h_0[i], c_0[i]))
                h_1.append(h_1_i), c_1.append(c_1_i)
                inp = h_1_i
                # only add dropout to hidden layer (not output)
                if i + 1 != self.num_layers and self.has_dropout:
                    inp = F.dropout(
                        inp, p=self.dropout, training=self.training)
            output, hidden = inp, (torch.stack(h_1), torch.stack(c_1))
        else:
            h_0, h_1 = hidden, []
            for i in range(self.num_layers):
                layer = getattr(self, self.cell + ('Cell_%d' % i))
                h_1_i = layer(inp, h_0[i])
                h_1.append(h_1_i)
                inp = h_1_i
                if i + 1 != self.num_layers and self.has_dropout:
                    inp = F.dropout(
                        inp, p=self.dropout, training=self.training)
            output, hidden = inp, torch.stack(h_1)
        return output, hidden


class MaxOut(nn.Module):
    def __init__(self, in_dim, out_dim, k):
        """
        Implementation of MaxOut:
            h_i^{maxout} = max_{j \in [1, ..., k]} x^T W_{..., i, j} + b_{i, j}
        where W is in R^{D x M x K}, D is the input size, M is the output size
        and K is the number of pieces to max-pool from. (i.e. i ranges over M,
        j ranges over K and ... corresponds to the input dimension)

        Parameters:
        ===========
        - in_dim: int, Input dimension
        - out_dim: int, Output dimension
        - k: int, number of "pools" to max over

        Returns:
        ===========
        - out: torch.Tensor (batch x k)
        """
        self.in_dim, self.out_dim, self.k = in_dim, out_dim, k
        super(MaxOut, self).__init__()
        self.projection = nn.Linear(in_dim, k * out_dim)

    def forward(self, inp):
        """
        Because of the linear projection we are bound to 1-d input
        (excluding batch-dim), therefore there is no need to generalize
        the implementation to n-dimensional input.
        """
        batch, in_dim = inp.size()
        # (batch x self.k * self.out_dim) -> (batch x self.out_dim x self.k)
        out = self.projection(inp).view(batch, self.out_dim, self.k)
        out, _ = out.max(2)
        return out.squeeze(2)


class TiedEmbedding(nn.Embedding):
    """
    Custom embeddings layer that takes an external weight parameter.
    This is useful for tying the embedding weights with another layer,
    typically the output vocabulary distribution (output embeddings).
    See TiedLinear.
    """
    def __init__(self, num_embeddings, embedding_dim, weight, **kwargs):
        super(TiedEmbedding, self).__init__(
            num_embeddings, embedding_dim, **kwargs)
        assert isinstance(weight, nn.parameter.Parameter)
        self.weight = weight


class TiedLinear(nn.Linear):
    """
    Custom linear layer that takes an external weight parameter.
    This is useful for tying the linear weights with another layer,
    typically the input vocabulary distribution (input embeddings).
    See TiedEmbedding.
    """
    def __init__(self, in_features, out_features, weight, bias=True):
        super(TiedLinear, self).__init__(in_features, out_features, bias=bias)
        assert isinstance(weight, nn.parameter.Parameter)
        self.weight = weight


class LM(nn.Module):
    """
    Vanilla RNN-based language model.

    Parameters:
    ===========
    - vocab: int, vocabulary size.
    - emb_dim: int, embedding size,
        This value has to be equal to hid_dim if tie_weights is True and
        project_on_tied_weights is False, otherwise input and output
        embedding dimensions wouldn't match and weights cannot be tied.
    - hid_dim: int, hidden dimension of the RNN.
    - num_layers: int, number of layers of the RNN.
    - cell: str, one of GRU, LSTM.
    - bias: bool, whether to include bias in the RNN.
    - dropout: float, amount of dropout to apply in between layers.
    - tie_weights: bool, whether to tie input and output embedding layers.
    - project_on_tied_weights: bool,
        In case of unequal emb_dim and hid_dim values this option has to
        be True if tie_weights is True. A linear project layer will be
        inserted after the RNN to match back to the embedding dimension.
    """
    def __init__(self, vocab, emb_dim, hid_dim, num_layers=1,
                 cell='GRU', bias=True, dropout=0.0, tie_weights=False,
                 project_on_tied_weights=False):
        self.vocab = vocab
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.tie_weights = tie_weights
        self.project_on_tied_weights = project_on_tied_weights
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
        # input embeddings
        weight = None
        if tie_weights:
            weight = nn.parameter.Parameter(torch.randn(vocab, emb_dim))
            self.embeddings = TiedEmbedding(vocab, self.emb_dim, weight)
        else:
            self.embeddings = nn.Embedding(vocab, self.emb_dim)
        # rnn
        self.rnn = getattr(nn, cell)(
            self.emb_dim, self.hid_dim,
            num_layers=num_layers, bias=bias, dropout=dropout)
        # output embeddings
        if not tie_weights:
            self.project = nn.Linear(self.hid_dim, vocab)
        else:
            if self.emb_dim == self.hid_dim:
                self.project = TiedLinear(self.hid_dim, vocab, weight)
            else:
                assert project_on_tied_weights, \
                    "Unequal tied layer dims but no projection layer"
                self.project = nn.Sequential(
                    nn.Linear(self.hid_dim, self.emb_dim),
                    TiedLinear(self.emb_dim, vocab, weight))

    def parameters(self):
        for p in super(LM, self).parameters():
            if p.requires_grad is True:
                yield p

    def n_params(self):
        return sum([p.nelement() for p in self.parameters()])

    def freeze_submodule(self, module):
        for p in getattr(self, module).parameters():
            p.requires_grad = False

    def init_hidden_for(self, inp):
        batch = inp.size(1)
        size = (self.num_layers, batch, self.hid_dim)
        h_0 = Variable(inp.data.new(*size).zero_(), requires_grad=False)
        if self.cell.startswith('GRU'):
            return h_0
        else:
            c_0 = Variable(inp.data.new(*size).zero_(), requires_grad=False)
            return h_0, c_0

    def generate_beam(self, bos, eos,
                      max_seq_len=20, width=5, gpu=False, **kwargs):
        "Generate text using beam search decoding"
        self.eval()
        beam = Beam(width, bos, eos, gpu=gpu)
        hidden = self.init_hidden_for(beam.get_current_state())
        while beam.active and len(beam) < max_seq_len:
            prev = Variable(
                beam.get_current_state().unsqueeze(0), volatile=True)
            logs, hidden = self(prev, hidden=hidden, **kwargs)
            beam.advance(logs.data)
            if self.cell.startswith('LSTM'):
                hidden = (u.swap(hidden[0], 1, beam.get_source_beam()),
                          u.swap(hidden[1], 1, beam.get_source_beam()))
            else:
                hidden = u.swap(hidden, 1, beam.get_source_beam())
        scores, hyps = beam.decode()
        return scores, hyps

    def generate(self, bos, eos,
                 max_seq_len=20, beam=None, gpu=False, **kwargs):
        "Generate text using simple argmax decoding"
        self.eval()
        prev = Variable(torch.LongTensor([bos]).unsqueeze(0), volatile=True)
        if gpu: prev = prev.cuda()
        hidden, hyp, score = None, [], []
        for _ in range(max_seq_len):
            logs, hidden = self(prev, hidden=hidden, **kwargs)
            best_score, prev = logs.max(1)
            prev = prev.t()
            hyp.append(prev), score.append(best_score)
            if prev.data.eq(eos).nonzero().nelement() > 0:
                break
        return [score], [hyp]

    def predict_proba(self, inp, gpu=False, **kwargs):
        self.eval()
        inp_vec = Variable(torch.LongTensor([inp]), volatile=True)
        if gpu:
            inp_vec.cuda()
        logs, hidden = self(inp_vec, **kwargs)
        return u.select_cols(logs, inp).sum().data[0]

    def forward(self, inp, hidden=None, **kwargs):
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


class ForkableLM(LM):
    """
    A LM model that allows to create forks of the current instance with
    frozen Embedding (and eventually RNN) layers for fine tunning the
    non-frozen parameters to particular dataset.
    The parent cannot have the projection layer for tied embeddings,
    since tied layers don't fit in this setup.
    """
    def __init__(self, *args, **kwargs):
        super(ForkableLM, self).__init__(*args, **kwargs)

    def fork_model(self, freeze_rnn=True):
        """
        Creates a child fork from the current model with frozen input
        embeddings (and eventually also frozen RNN).

        Parameters:
        ===========
        - freeze_rnn: optional, whether to also freeze the child RNN layer.
        """
        model = ForkableLM(
            self.vocab, self.emb_dim, self.hid_dim, num_layers=self.num_layers,
            cell=self.cell, dropout=self.dropout, bias=self.bias,
            tie_weights=False, project_on_tied_weights=False)
        source_dict, target_dict = self.state_dict(), model.state_dict()
        target_dict['embeddings.weight'] = source_dict()['embeddings.weight']
        for layer, p in source_dict.items():
            if layer.startswith('project') and \
               self.tie_weights and \
               self.project_on_tied_weights:
                print("Warning: Forked model couldn't use projection layer " +
                      "of parent for the initialization of layer [%s]" % layer)
                continue
            else:
                target_dict[layer] = p
        model.load_state_dict(target_dict)
        model.freeze_submodule('embeddings')
        if freeze_rnn:
            model.freeze_submodule('rnn')
        return model


class MultiheadLM(LM):
    """
    A variant LM that has multiple output embeddings (one for each of a
    given number of heads). This allows the model to fine tune different
    output distribution on different datasets.
    """
    def __init__(self, vocab, emb_dim, hid_dim, num_layers=1,
                 cell='GRU', bias=True, dropout=0.0, heads=(), **kwargs):
        assert heads, "MultiheadLM requires at least 1 head but got 0"
        self.vocab = vocab
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.num_layers = num_layers
        self.cell = cell
        self.bias = bias
        self.has_dropout = bool(dropout)
        self.dropout = dropout
        self.heads = heads

        super(LM, self).__init__()
        self.embeddings = nn.Embedding(vocab, self.emb_dim)
        self.rnn = getattr(nn, cell)(
            self.emb_dim, self.hid_dim,
            num_layers=num_layers, bias=bias, dropout=dropout)
        self.project = {}
        for head in heads:
            module = nn.Linear(self.hid_dim, vocab)
            self.add_module(head, module)
            self.project[head] = module

    def forward(self, inp, hidden=None, head=None):
        """"""
        emb = self.embeddings(inp)
        if self.has_dropout:
            emb = F.dropout(emb, p=self.dropout, training=self.training)
        outs, hidden = self.rnn(emb, hidden or self.init_hidden_for(emb))
        if self.has_dropout:
            outs = F.dropout(outs, p=self.dropout, training=self.training)
        seq_len, batch, hid_dim = outs.size()
        # (seq_len x batch x hid) -> (seq_len * batch x hid)
        logs = self.project[head](outs.view(seq_len * batch, hid_dim))
        return logs, hidden


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


class LMContainer(object):
    def __init__(self, models, d):
        """
        Constructor

        Parameters:
        ===========
        - models, a dict mapping from head names to models or a MultiheadLM
        - d, a Dict or a dict mapping from head names to Dict's
        """
        self.models = models
        self.d = d
        if isinstance(self.models, dict):
            for model in self.models.values():
                assert isinstance(model, ForkableLM), "Expected ForkableLM"
            # forkable models
            self.heads = list(d.keys())
            self.get_head = lambda head: self.models[head]
        elif isinstance(self.models, MultiheadLM):
            self.heads = list(models.heads)
            self.get_head = lambda head: self.models
        else:
            raise ValueError("Wrong model type %s" % type(models))

    def cuda(self):
        for head in self.heads:
            self.get_head(head).cuda()

    def cpu(self):
        for head in self.heads:
            self.get_head(head).cpu()

    def predict_proba(self, text, author, gpu=False):
        inp = [c for l in self.d.transform(text) for c in l]
        return self.get_head(author).predict_proba(inp, head=author)

    def to_disk(self, prefix, mode='torch'):
        if mode == 'torch':
            import torch
            save_fn, ext = torch.save, 'pt'
        elif mode == 'pickle':
            import pickle as p
            save_fn, ext = p.dump, 'pickle'
        else:
            raise ValueError("Unknown mode [%s]" % mode)
        if isinstance(self.models, dict):
            # forkable models
            for head, model in self.models.items():
                with open(prefix + '_' + head + '.' + ext, 'wb') as f:
                    save_fn(model, f)
        else:
            with open(prefix + '.' + ext, 'wb') as f:
                save_fn(self.models, f)
        with open(prefix + '.dict.' + ext, 'wb') as f:
            save_fn(self.d, f)

    @classmethod
    def from_disk(cls, model_path, d_path):
        """
        Parameters:
        ===========

        - model_path: str,
            Path to file with serialized MultiheadLM model, or dict from
            heads to paths with ForkableLM models.
        - d_path: str,
            Path to file with serialized Dict.
        """
        if isinstance(model_path, dict):
            model = {}
            for k, path in model_path.items():
                model[k] = load_model(path)
        else:
            model = load_model(model_path)
        d = load_model(d_path)
        return cls(model, d)

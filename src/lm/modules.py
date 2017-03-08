
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        -----------
        inp: torch.Tensor (batch x inp_dim),
            Tensor holding the target for the previous decoding step,
            inp_dim = emb_dim or emb_dim + hid_dim if self.add_pred is True.
        hidden: tuple (h_c, c_0), output of previous step or init hidden at 0,
            h_c: (num_layers x batch x hid_dim)
            n_c: (num_layers x batch x hid_dim)
        Returns: output, (h_n, c_n)
        --------
        output: torch.Tensor (batch x hid_dim)
        h_n: torch.Tensor (num_layers x batch x hid_dim)
        c_n: torch.Tensor (num_layers x batch x hid_dim)
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
        -----------
        in_dim: int, Input dimension
        out_dim: int, Output dimension
        k: int, number of "pools" to max over

        Returns:
        --------
        out: torch.Tensor (batch x k)
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
    def __init__(self, num_embeddings, embedding_dim, weight, **kwargs):
        super(TiedEmbedding, self).__init__(
            num_embeddings, embedding_dim, **kwargs)
        assert isinstance(weight, nn.parameter.Parameter)
        self.weight = weight


class TiedLinear(nn.Linear):
    def __init__(self, in_features, out_features, weight, bias=True):
        super(TiedLinear, self).__init__(in_features, out_features, bias=bias)
        assert isinstance(weight, nn.parameter.Parameter)
        self.weight = weight


class LM(nn.Module):
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
                    TiedLinear(self.emb_dim, vocab, weight))
        else:
            self.project = nn.Linear(self.hid_dim, vocab)

    def parameters(self):
        for p in super(LM, self).parameters():
            if p.requires_grad is True:
                yield p

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


class ForkableLM(LM):
    def __init__(self, *args, **kwargs):
        super(LM, self).__init__(*args, **kwargs)

    def fork_model(self):
        model = LM(self.vocab, self.emb_dim, self.hid_dim,
                   num_layers=self.num_layers, cell=self.cell, bias=self.bias,
                   dropout=self.dropout, tie_weights=False,
                   project_on_tied_weights=False)
        state_dict = model.state_dict()
        state_dict['embeddings.weight'] = self.state_dict()['embeddings.weight']
        for layer, p in self.state_dict().items():
            if layer.startswith('project') and \
               self.tie_weights and \
               self.project_on_tied_weights:
                print("Warning: Forked model couldn't use projection layer of" +
                      "parent for the initialization of layer [%s]" % layer)
                continue
            else:
                state_dict[layer] = p
        model.load_state_dict(state_dict)
        model.freeze_submodule('embeddings')
        model.freeze_submodule('rnn')
        return model

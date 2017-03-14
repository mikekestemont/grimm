
import random
from collections import Counter

import torch
from torch.autograd import Variable


def shuffled(data):
    data = list(data)
    random.shuffle(data)
    return data


def shuffle_pairs(pair1, pair2):
    return zip(*shuffled(zip(pair1, pair2)))


def cumsum(seq):
    seq = [0] + list(seq)
    subseqs = (seq[:i] for i in range(1, len(seq)+1))
    return [sum(subseq) for subseq in subseqs]


def get_splits(length, test, dev=None):
    splits = [split for split in [dev, test] if split]
    train = 1 - sum(splits)
    assert train > 0, "dev and test proportions must add to at most 1"
    return cumsum(int(length * i) for i in [train, dev, test] if i)


def batchify(examples, pad_token=None, align_right=False):
    max_length = max(len(x) for x in examples)
    out = torch.LongTensor(len(examples), max_length).fill_(pad_token or 0)
    for i in range(len(examples)):
        example = torch.Tensor(examples[i])
        example_length = example.size(0)
        offset = max_length - example_length if align_right else 0
        out[i].narrow(0, offset, example_length).copy_(example)
    out = out.t().contiguous()
    return out


def block_batchify(vector, batch_size):
    if isinstance(vector, list):
        vector = torch.LongTensor(vector)
    num_batches = len(vector) // batch_size
    batches = vector.narrow(0, 0, num_batches * batch_size)
    batches = batches.view(batch_size, -1).t().contiguous()
    return batches


class Dict(object):
    def __init__(self, pad_token=None, eos_token=None, bos_token=None,
                 unk_token='<unk>', max_size=None, min_freq=1,
                 sequential=True):
        """
        Dict
        """
        self.counter = Counter()
        self.vocab = [t for t in [pad_token, eos_token, bos_token] if t]
        self.fitted = False
        self.has_unk = False    # only index unk_token if needed
        self.pad_token = pad_token
        self.eos_token = eos_token
        self.bos_token = bos_token
        self.unk_token = unk_token
        self.max_size = max_size
        self.min_freq = min_freq
        self.sequential = sequential

    def __len__(self):
        return len(self.vocab)

    def get_pad(self):
        return self.s2i.get(self.pad_token, None)

    def get_eos(self):
        return self.s2i.get(self.eos_token, None)

    def get_bos(self):
        return self.s2i.get(self.bos_token, None)

    def get_unk(self):
        return self.s2i.get(self.unk_token, None)

    def _maybe_index_unk(self):
        if self.unk_token not in self.s2i:
            unk_code = self.s2i[self.unk_token] = len(self.vocab)
            self.vocab += [self.unk_token]
            self.has_unk = True
            return unk_code
        else:
            return self.s2i[self.unk_token]

    def index(self, s):
        assert self.fitted, "Attempt to index without fitted data"
        if s not in self.s2i:
            return self._maybe_index_unk()
        else:
            return self.s2i[s]

    def partial_fit(self, *args):
        for dataset in args:
            for example in dataset:
                self.counter.update(example if self.sequential else [example])

    def fit(self, *args):
        self.partial_fit(*args)
        most_common = self.counter.most_common(self.max_size)
        self.vocab += [k for k, v in most_common if v >= self.min_freq]
        self.s2i = {s: i for i, s in enumerate(self.vocab)}
        self.fitted = True
        return self

    def transform(self, examples, bos=True, eos=True):
        bos = [self.index(self.bos_token)] if self.bos_token and bos else []
        eos = [self.index(self.eos_token)] if self.eos_token and eos else []
        for example in examples:
            if self.sequential:
                example = bos + [self.index(s) for s in example] + eos
            else:
                example = self.index(example)
            yield example


class Dataset(object):
    def __init__(self, src, trg, dicts, fitted=False):
        """
        Constructs a dataset out of source and target pairs. Examples will
        be transformed into integers according to their respective Dict.
        The required dicts can be references to the same Dict instance in
        case of e.g. monolingual data.

        Arguments:
        - src: list of lists of hashables representing source sequences
        - trg: list of lists of hashables representing target sequences
        - dicts: dict of {'src': src_dict, 'trg': trg_dict} where
            src_dict: Dict instance fitted to the source data
            trg_dict: Dict instance fitted to the target data
        - sort_key: function to sort src, trg example pairs
        """
        self.src = src if fitted else list(dicts['src'].transform(src))
        self.trg = trg if fitted else list(dicts['trg'].transform(trg))
        assert len(src) == len(trg), \
            "Source and Target dataset must be equal length"
        self.dicts = dicts      # fitted dicts

    def __len__(self):
        return len(self.src)

    def sort_(self, sort_key=None):
        src, trg = zip(*sorted(zip(self.src, self.trg), key=sort_key))
        self.src, self.trg = src, trg
        return self

    def batches(self, batch_size, **kwargs):
        """
        Returns a BatchIterator built from this dataset examples

        Parameters:
        - batch_size: Integer
        - kwargs: Parameters passed on to the BatchIterator constructor
        """
        total_len = len(self.src)
        assert batch_size <= total_len, \
            "Batch size larger than data [%d > %d]" % (batch_size, total_len)
        return BatchIterator(self, batch_size, **kwargs)

    def splits(self, test=0.1, dev=0.2, shuffle=False,
               batchify=False, batch_size=None, sort_key=None, **kwargs):
        """
        Compute splits on dataset instance. For convenience, it can return
        BatchIterator objects instead of Dataset via method chaining.

        Parameters:
        ===========
        - dev: float less than 1 or None, dev set proportion
        - test: float less than 1 or None, test set proportion
        - shuffle: bool, whether to shuffle the datasets prior to splitting
        - batchify: bool, whether to return BatchIterator's instead
        - batch_size: int, only needed if batchify is True
        - kwargs: optional arguments passed to the BatchIterator constructor
        """
        if shuffle:
            src, trg = shuffle_pairs(self.src, self.trg)
        else:
            src, trg = self.src, self.trg
        splits = get_splits(len(src), test, dev=dev)
        datasets = [Dataset(src[i:j], trg[i:j], self.dicts, fitted=True)
                    .sort_(sort_key) for i, j in zip(splits, splits[1:])]
        if batchify:
            return tuple([s.batches(batch_size, **kwargs) for s in datasets])
        else:
            return datasets

    @classmethod
    def from_disk(cls, path):
        data = torch.load(path)
        src, trg, dicts = data['src'], data['trg'], data['dicts']
        return cls(src, trg, dicts, fitted=True)

    def to_disk(self, path):
        data = {'src': self.src, 'trg': self.trg, 'dicts': self.dicts}
        torch.save(data, path)


class BatchIterator(object):
    def __init__(self, dataset, batch_size,
                 gpu=False, align_right=False, evaluation=False):
        """
        BatchIterator
        """
        self.dataset = dataset
        self.src = dataset.src
        self.trg = dataset.trg
        self.batch_size = batch_size
        self.gpu = gpu
        self.align_right = align_right
        self.evaluation = evaluation
        self.num_batches = len(dataset) // batch_size

    def _batchify(self, batch_data, pad_token):
        out = batchify(batch_data, pad_token, align_right=self.align_right)
        if self.gpu:
            out = out.cuda()
        return Variable(out, volatile=self.evaluation)

    def __getitem__(self, idx):
        assert idx < self.num_batches, "%d >= %d" % (idx, self.num_batches)
        batch_from = idx * self.batch_size
        batch_to = (idx+1) * self.batch_size
        src_pad = self.dataset.dicts['src'].get_pad()
        trg_pad = self.dataset.dicts['trg'].get_pad()
        src_batch = self._batchify(self.src[batch_from: batch_to], src_pad)
        trg_batch = self._batchify(self.trg[batch_from: batch_to], trg_pad)
        return src_batch, trg_batch

    def __len__(self):
        return self.num_batches


class BlockDataset(object):
    """
    Dataset class for training LMs that also supports multi-source datasets.

    Parameters:
    ===========
    - examples: list of sequences or dict of source to list of sequences,
        Source data that will be used by the dataset.
        If fitted is False, the lists are supposed to be already transformed
        into a single long vector. If a dict, the examples are supposed to
        come from different sources and will be iterated over cyclically.
    - src_dict: Dict already fitted.
    - batch_size: int,
    - bptt: int,
        Backpropagation through time (maximum context that the RNN should pay
        attention to)
    """
    def __init__(self, examples, src_dict, batch_size, bptt,
                 fitted=False, gpu=False, evaluation=False):
        if isinstance(examples, dict):
            self.data = {}
            for name, data in examples.items():
                if not fitted:      # subdata is already an integer vector
                    data = [c for l in src_dict.transform(data) for c in l]
                self.data[name] = block_batchify(data, batch_size)
            self.names = list(self.data.keys())
        else:
            if not fitted:
                examples = \
                    [c for l in src_dict.transform(examples) for c in l]
            self.data = block_batchify(examples, batch_size)

        self.src_dict = src_dict
        self.batch_size = batch_size
        self.bptt = bptt
        self.fitted = fitted
        self.gpu = gpu
        self.evaluation = evaluation

    def _next_item(self, idx):
        """
        Selects next dataset in case of multiple datasets in a cyclical way.
        """
        idx, dataset = divmod(idx, len(self.data))
        name = self.names[dataset]
        data = self.data[name]
        return idx, name, data

    def __getitem__(self, idx):
        if isinstance(self.data, dict):
            idx, name, data = self._next_item(idx)
        else:
            data = self.data
        idx *= self.bptt
        seq_len = min(self.bptt, len(data) - 1 - idx)
        src = Variable(data[idx:idx+seq_len], volatile=self.evaluation)
        trg = Variable(data[idx+1:idx+seq_len+1].view(-1),
                       volatile=self.evaluation)
        if self.gpu:
            src, trg = src.cuda(), trg.cuda()
        if isinstance(self.data, dict):
            return src, trg, name
        else:
            return src, trg

    def __len__(self):
        """
        The length of the dataset is computed as the number of bptt'ed batches
        to conform the way batches are computed. See __getitem__.
        """
        if isinstance(self.data, dict):
            num_batches = min(len(d) for d in self.data.values()) * len(self.data)
        else:
            num_batches = len(self.data)
        return num_batches // self.bptt

    def splits(self, test=0.1, dev=0.1):
        """
        Computes splits according to test and dev proportions (whose sum can't
        be higher than 1). In case of a multi-source dataset, the output is
        respectively a dataset containing the partition for each source in the
        same shape as the original (non-partitioned) dataset.

        Returns:
        ==========

        tuple of BlockDataset's
        """
        n_element = len(self) * self.bptt * self.batch_size  # min num elements
        datasets, splits = [], get_splits(n_element, test, dev=dev)
        for idx, (start, stop) in enumerate(zip(splits, splits[1:])):
            evaluation = self.evaluation if idx == 0 else True
            if isinstance(self.data, dict):
                start, stop = start // len(self.data), stop // len(self.data)
                dataset = {}
                for name, data in self.data.items():
                    dataset[name] = data.t().contiguous().view(-1)[start:stop]
                datasets.append(BlockDataset(
                    dataset, self.src_dict, self.batch_size, self.bptt,
                    fitted=True, gpu=self.gpu, evaluation=evaluation))
            else:
                datasets.append(BlockDataset(
                    self.data[start:stop], self.src_dict, self.batch_size,
                    self.bptt, fitted=True, gpu=self.gpu,
                    evaluation=evaluation))
        return tuple(datasets)


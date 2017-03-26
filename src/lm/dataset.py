
import random
from collections import Counter

import torch
import torch.utils.data
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
        if self.fitted:
            raise ValueError('Dict is already fitted')
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


class BatchIterator(object):
    """
    BatchIterator, a class to batchify PairedDataset's
    """
    def __init__(self, dataset, batch_size,
                 gpu=False, align_right=False, evaluation=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.gpu = gpu
        self.evaluation = evaluation
        self.align_right = align_right
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
        src_pad = self.dataset.d['src'].get_pad()
        trg_pad = self.dataset.d['trg'].get_pad()
        src_batch = self._batchify(
            self.dataset.data['src'][batch_from: batch_to], src_pad)
        trg_batch = self._batchify(
            self.dataset.data['trg'][batch_from: batch_to], trg_pad)
        return src_batch, trg_batch

    def __len__(self):
        return self.num_batches


class Dataset(torch.utils.data.Dataset):
    @classmethod
    def from_disk(cls, path):
        with open(path, 'rb') as f:
            return torch.load(f)

    def to_disk(self, path):
        with open(path, 'wb') as f:
            torch.save(self, f)


class PairedDataset(Dataset):
    def __init__(self, src, trg, d, fitted=False):
        """
        Constructs a dataset out of source and target pairs. Examples will
        be transformed into integers according to their respective Dict.
        The required dicts can be references to the same Dict instance in
        case of e.g. monolingual data.

        Arguments:
        - src: list of lists of hashables representing source sequences
        - trg: list of lists of hashables representing target sequences
        - d: dict of {'src': src_dict, 'trg': trg_dict} where
            src_dict: Dict instance fitted to the source data
            trg_dict: Dict instance fitted to the target data
        """
        self.data = {
            'src': src if fitted else list(d['src'].transform(src)),
            'trg': trg if fitted else list(d['trg'].transform(trg))}
        assert len(src) == len(trg), \
            "Source and Target dataset must be equal length"
        self.d = d              # fitted dicts

    def __len__(self):
        return len(self.data['src'])

    def sort_(self, sort_key=None):
        sort = sorted(zip(self.data['src'], self.data['trg']), key=sort_key)
        src, trg = zip(*sort)
        self.data['src'] = src,
        self.data['trg'] = trg
        return self

    def batches(self, batch_size, **kwargs):
        """
        Returns a BatchIterator built from this dataset examples

        Parameters:
        - batch_size: Integer
        """
        total_len = len(self)
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
        """
        if shuffle:
            src, trg = shuffle_pairs(self.data['src'], self.data['trg'])
        else:
            src, trg = self.data['src'], self.data['trg']
        splits, datasets = get_splits(len(src), test, dev=dev), []
        for i, j in zip(splits, splits[1:]):
            subset = PairedDataset(
                src[i:j], trg[i:j], self.d, fitted=True).sort_(sort_key)
            datasets.append(subset)
        if batchify:
            return tuple([s.batches(batch_size, **kwargs) for s in datasets])
        else:
            return datasets


class BlockDataset(Dataset):
    """
    Dataset class for training LMs that also supports multi-source datasets.

    Parameters:
    ===========
    - examples: list of sequences or dict of source to list of sequences,
        Source data that will be used by the dataset.
        If fitted is False, the lists are supposed to be already transformed
        into a single long vector. If a dict, the examples are supposed to
        come from different sources and will be iterated over cyclically.
    - d: Dict already fitted.
    - batch_size: int,
    - bptt: int,
        Backpropagation through time (maximum context that the RNN should pay
        attention to)
    """
    def __init__(self, examples, d, batch_size, bptt,
                 fitted=False, gpu=False, evaluation=False):
        if not fitted:
            examples = [c for l in d.transform(examples) for c in l]
        self.data = block_batchify(examples, batch_size)
        if len(examples) == 0:
            raise ValueError("Empty dataset")

        self.d = d
        self.batch_size = batch_size
        self.bptt = bptt
        self.fitted = fitted
        self.gpu = gpu
        self.evaluation = evaluation

    def __len__(self):
        """
        The length of the dataset is computed as the number of bptt'ed batches
        to conform the way batches are computed. See __getitem__.
        """
        return len(self.data) // self.bptt

    def _getitem(self, data, idx):
        """
        General function to get the source data to compute the batch. This
        should be overwritten by subclasses in which the source data isn't
        always stored in self.data, e.g. the case of cyclical subset access.
        """
        idx *= self.bptt
        seq_len = min(self.bptt, len(data) - 1 - idx)
        src = Variable(data[idx:idx+seq_len], volatile=self.evaluation)
        trg = Variable(data[idx+1:idx+seq_len+1].view(-1),
                       volatile=self.evaluation)
        if self.gpu:
            src, trg = src.cuda(), trg.cuda()
        return src, trg

    def __getitem__(self, idx):
        """
        Item getter for batch number idx.

        Returns:
        ========
        - src: torch.LongTensor of maximum size of self.bptt x self.batch_size
        - trg: torch.LongTensor of maximum size of self.bptt x self.batch_size,
            corresponding to a shifted batch
        """
        return self._getitem(self.data, idx)

    def split_data(self, start, stop):
        """
        Compute a split on the dataset for a batch range defined by start, stop
        """
        return self.data.t().contiguous().view(-1)[start:stop]

    def splits(self, test=0.1, dev=0.1):
        """
        Computes splits according to test and dev proportions (whose sum can't
        be higher than 1). In case of a multi-source dataset, the output is
        respectively a dataset containing the partition for each source in the
        same shape as the original (non-partitioned) dataset.

        Returns:
        ========

        tuple of BlockDataset's
        """
        n_element = len(self) * self.bptt * self.batch_size  # min num elements
        datasets, splits = [], get_splits(n_element, test, dev=dev)
        for idx, (start, stop) in enumerate(zip(splits, splits[1:])):
            evaluation = self.evaluation if idx == 0 else True
            split = self.split_data(start, stop)
            datasets.append(type(self)(
                split, self.d, self.batch_size, self.bptt,
                fitted=True, gpu=self.gpu, evaluation=evaluation))
        return tuple(datasets)


class CyclicBlockDataset(BlockDataset):
    def __init__(self, examples, d, batch_size, bptt,
                 fitted=False, gpu=False, evaluation=False):
        self.data = {}
        for name, data in examples.items():
            if not fitted:      # subdata is already an integer vector
                data = [c for l in d.transform(data) for c in l]
            self.data[name] = block_batchify(data, batch_size)

        self.names = list(self.data.keys())
        self.d = d
        self.batch_size = batch_size
        self.bptt = bptt
        self.fitted = fitted
        self.gpu = gpu
        self.evaluation = evaluation

    def __len__(self):
        batches = min(len(d) for d in self.data.values()) * len(self.data)
        return batches // self.bptt

    def __getitem__(self, idx):
        """
        Computes the subset for each batch in a cyclical way.

        Returns:
        ========
        In addition to the standard BlockDataset, this subclass also returns
        the name of the dataset in the current cycle in third position.
        """
        idx, dataset = divmod(idx, len(self.data))
        name = self.names[dataset]
        data = self.data[name]
        src, trg = self._getitem(data, idx)
        return src, trg, name

    def split_data(self, start, stop):
        start, stop = start // len(self.data), stop // len(self.data)
        split = {}
        for name, data in self.data.items():
            split[name] = data.t().contiguous().view(-1)[start:stop]
        return split

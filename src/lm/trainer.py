
import time
import math
import torch
from torch.autograd import Variable


# Utility functions (repackage_hidden, memory effective loss, etc.)
def repackage_hidden(h):
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


# Loggers
class Logger(object):
    @staticmethod
    def epoch_begin(payload):
        print("Starting epoch [%d]" % payload['epoch'])

    @staticmethod
    def epoch_end(epoch, payload):
        print("Epoch [%d], train loss: %g. Took: %ds" %
              (payload['epoch'], payload["loss"], payload["duration"]))

    @staticmethod
    def validation_begin(payload):
        print("Epoch [%d], valid loss: %g" % (payload['epoch'], payload['loss']))

    @staticmethod
    def test_end(payload):
        print("Test loss: %g" % payload["loss"])
        
    def log(self, event, payload, verbose=True):
        if verbose:
            getattr(self, event)(payload)


# Base Trainer class
class Trainer(object):
    def __init__(self, model, d, datasets, criterion, optimizer, verbose=True):
        # attributes
        self.model = model
        self.d = d
        self.datasets = datasets   # is a dict
        self.criterion = criterion  # might be a dict
        self.optimizer = optimizer  # might be a dict
        # config
        self.verbose = verbose
        # containers
        self.hooks = []
        self.loggers = []
        self.locals = {}
        # properties
        self.test_name = 'test'
        self.valid_name = 'valid'

    @property
    def test_name(self):
        return self.test_name

    @property
    def valid_name(self):
        return self.valid_name

    # logging
    def add_loggers(self, *loggers):
        for logger in loggers:
            self.loggers.append(logger)

    def log(self, event, payload):
        for logger in self.loggers:
            logger.log(event, payload, verbose=self.verbose)

    def on_epoch_begin(self, epoch):
        self.log("epoch_begin", {"epoch": epoch})

    def on_epoch_end(self, epoch, loss, duration):
        self.log("epoch_end", {"epoch": epoch,
                               "loss": self.format_loss(loss),
                               "duration": duration})

    def on_validation_end(self, epoch, loss):
        self.log("validation_end", {"epoch": epoch, "loss": self.format_loss(loss)})

    def on_test_end(self, loss):
        self.log("test_end", {"loss": self.format_loss(loss)})

    # hooks
    def add_hooks(self, hook, num_checkpoints=1):
        self.hooks.add({'hook': hook, 'num_checkpoints': num_checkpoints})

    def run_hooks(self, batch_num, checkpoint):
        for hook in self.hooks:
            checkpoint = (batch_num // checkpoint)
            if checkpoint % hook['num_checkpoints'] == 0:
                hook['hook'](batch_num, checkpoint)

    # reporting
    def format_loss(self, loss):
        return loss

    def num_batch_examples(self, batch_data):
        return batch_data.n_element()

    # optimizer
    def zero_grad(self):
        if isinstance(self.optimizer, dict):
            for opt in self.optimizer.values():
                opt.zero_grad()
        else:
            self.optimizer.zero_grad()

    def optimizer_step(self):
        if isinstance(self.optimizer, dict):
            for opt in self.optimizer.values():
                opt.step()
        else:
            self.optimizer.step()

    # training code
    def on_batch_end(self, batch, batch_loss):
        # reset hidden, and things like that
        pass

    def validate_model(self, **kwargs):
        loss, num_words = 0, 0
        dataset = self.datasets[self.get_valid_name()]
        for batch in range(len(dataset)):
            loss += self.run_batch(
                batch, dataset=self.get_valid_name(), **kwargs)
            num_words += self.num_batch_examples(dataset[batch])
        return loss.data[0] / num_words

    def run_batch(self, batch, dataset='train', **kwargs):
        source, targets = self.datasets[dataset][batch]
        outs = self.model(source, targets)
        loss = self.criterion(outs, targets)
        if dataset == 'train':
            loss.backward(), self.optimizer_step()
        return loss

    def train_epoch(self, epoch, checkpoint=1, shuffle=False, **kwargs):
        # compute batch order
        batch_order = range(len(self.datasets['train']))
        if shuffle:
            batch_order = np.random.permutation(batch_order)
        start = time.time()
        epoch_loss, checkpoint_loss, epoch_words, checkpoint_words = 0, 0, 0, 0
        for batch_num, batch in enumerate(batch_order):
            self.zero_grad()
            loss = self.run_batch(batch, dataset='train', **kwargs)
            self.on_batch_end(batch, loss)
            # report
            num_examples = self.num_batch_examples(self.datasets['train'][batch])
            epoch_loss += num_examples * loss.data[0]
            report_loss += num_examples * loss.data[0]
            epoch_words += num_examples
            report_words += num_examples
            # checkpoint
            if batch_num % checkpoint == 0 and batch_num > 0:
                self.log('checkpoint', {
                    'batch': batch_num,
                    'total_batches': len(batch_order),
                    'examples': num_examples,
                    'duration': time.time() - start,
                    'loss': self.format_loss(report_loss / report_words)})
                report_loss, report_words, start = 0, 0, time.time()
                # run hooks after checkpoint
                self.run_hooks(self, batch_num, checkpoint)
        return epoch_loss / epoch_words

    def train(self, epochs, checkpoint, gpu=False, early_stop=0, **kwargs):
        for epoch in range(1, epochs + 1):
            self.on_epoch_begin(epoch)
            start = time.time()
            # train
            self.model.train()
            train_loss = self.train_epoch(epoch, checkpoint, **kwargs)
            epoch_time = time.time() - start
            # valid
            if self.get_valid_name() in self.datasets:
                self.model.eval()
                valid_loss = self.validate_model(**kwargs)
                self.on_validation_end(epoch, valid_loss)
            # epoch end
            self.on_epoch_end(epoch, train_loss, epoch_time)
        # test
        if self.get_test_name() in self.datasets:
            self.model.eval()
            test_loss = self.validate_model(**kwargs)
            self.on_test_end(test_loss)


class LMTrainer(Trainer):
    def format_loss(self, loss):
        return math.exp(min(loss, 100))

    def batch_loss(self, batch, dataset='train', subset=None, **kwargs):
        data = self.datasets[dataset]
        hidden = self.locals.get('hidden', None)
        if isinstance(data, CyclicBlockDataset):
            source, targets, head = data[batch]
            if subset is not None and subset != head:
                # if subset is given, skip all other subsets
                continue
            output, hidden = self.model(source, hidden=hidden, head=head)
        else:
            source, targets = data[batch]
            output, hidden = self.model(source, hidden=hidden)
        loss = self.criterion(output, targets)
        if dataset == 'train':
            loss.backward(), self.optimizer_step()
        self.locals['hidden'] = repackage_hidden(hidden)
        return loss

    def on_batch_end(self, batch, loss):
        if getattr(self, 'reset_hidden'):
            self.locals['hidden'].zero_()

    def num_batch_examples(self, batch_data):
        return len(batch_data[0])


from lm import LM, MultiheadLM
import utils as u


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
                assert isinstance(model, LM), "Expected LM"
            # LM or ForkableLM models
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
        return self

    def cpu(self):
        for head in self.heads:
            self.get_head(head).cpu()
        return self

    def predict_proba(self, text, author, gpu=False):
        if isinstance(self.d, dict):
            d = self.d[author]
        else:
            d = self.d
        inp = [c for l in d.transform(text) for c in l]
        return self.get_head(author).predict_proba(inp, head=author)

    def to_disk(self, prefix, mode='torch'):
        self.cpu()              # always move to cpu
        if isinstance(self.models, dict):
            # LM models
            for head, model in self.models.items():
                u.save_model(prefix + '_' + head, mode=mode)
        else:
            u.save_model(self.models, prefix, mode=mode)
        u.save_model(self.d, prefix + '.dict', mode=mode)

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
                model[k] = u.load_model(path)
        else:
            model = u.load_model(model_path)
        d = u.load_model(d_path)
        return cls(model, d)

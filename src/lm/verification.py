
import numpy as np

from modules import ForkableLM, MultiheadLM


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

    def predict_proba(self, text, author, gpu=False):
        inp = [c for l in self.d.transform(text) for c in l]
        return self.get_head(author).predict_proba(inp, head=author)

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


class Attributor(object):
    def __init__(self, model_container):
        """
        Constructor

        Parameters:
        ===========
            - model, a language model container for several authors
        """
        self.model_container = model_container

    def predict_probas(self, texts):
        """
        Parameters:
        ===========
            - an iterable of strings to be attributed

        Returns:
        ===========
            - np.array with probability scores of shape (texts x authors)
        """
        text_probas = []

        for text in texts:
            text_probas.append(
                [self.model_container.predict_proba(text, author)
                 for author in self.model_container.heads])
        return np.array(text_probas, dtype=np.float32)

    def predict(self, texts):
        """
        Parameters:
        ===========
            - an iterable of strings to be attributed

        Returns:
        ===========
            - a list of strings with the author attribution
              for each text.
        """
        probas = self.get_probas(texts)
        return [self.model_container.heads[idx]
                for idx in probas.argmax(axis=-1)]

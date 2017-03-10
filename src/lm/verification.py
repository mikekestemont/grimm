
import numpy as np

from modules import ForkableLM, MultiheadLM


class LMContainer(object):
    def __init__(self, models, d):
        """
        Constructor

        Parameters:
        ===========
        - models, a dict mapping from head names to models
        - d, a Dict or a dict mapping from head names to Dict's
        """
        self.models = models
        self.d = d
        if isinstance(self.models, dict):
            for model in self.models.values():
                assert isinstance(model, ForkableLM), "Expected ForkableLM"
            # forkable models
            self.heads = d.keys()
            self.get_head = lambda head: self.models[head]
        elif isinstance(self.models, MultiheadLM):
            self.heads = models.heads
            self.get_head = lambda head: self.models
        else:
            raise ValueError("Wrong model type %s" % type(models))

    def predict_proba(self, text, author, gpu=False):
        inp = [c for l in self.d.transform(text) for c in l]
        return self.get_head(author).predict_proba(inp, head=author)


class Attributor(object):
    def __init__(self, model, d):
        """
        Constructor

        Parameters:
        ===========
            - model, a language model container for several authors
        """
        self.model = model
        self.d = d

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
            text_probas.append([self.model.predict_proba(text, author)
                                for author in self.model.heads])
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
        return [self.authors[idx] for idx in probas.argmax(axis=-1)]

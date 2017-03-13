
import numpy as np

from modules import ForkableLM, MultiheadLM


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


if __name__ == '__main__':
    import argparse
    from dataset import BlockDataset, Dict
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path')
    parser.add_argument('--corpus_path')

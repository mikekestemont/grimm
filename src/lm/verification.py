
import sys
import os
import numpy as np
from sklearn.metrics import accuracy_score

from modules import ForkableLM, MultiheadLM, LMContainer
from dataset import Dict


class Attributor(object):
    def __init__(self, model_container):
        """
        Constructor

        Parameters:
        ===========
            - model, a language model container for several authors
        """
        self.model_container = model_container
        self.authors = {author: idx for idx, author in enumerate(self.model_container.heads)}

    def predict_probas(self, texts):
        """
        Parameters:
        ===========
            - an iterable of docs (each doc being an iterable of sents)
              to be attributed

        Returns:
        ===========
            - np.array with probability scores of shape (texts x authors)
        """
        text_probas = []

        for idx, text in enumerate(texts):
            print("Processing [%d/%d]" % (idx + 1, len(texts)))
            text_probas.append(
                [self.model_container.predict_proba(text, author)
                 for author in self.model_container.heads])
        return np.array(text_probas, dtype=np.float32)

    def get_label(self, label):
        return self.model_container.heads[label]

    def predict(self, texts):
        """
        Parameters:
        ===========
            - an iterable of docs (each doc being an iterable of sents)
              to be attributed

        Returns:
        ===========
            - a list of ids with the author attribution for each text.
        """
        return self.predict_probas(texts).argmax(axis=-1)


def crop_letters(letters, min_len):
    for l in letters:
        lines, num_chars = [], 0
        for line in l.lines:
            get_chars = min(min_len - num_chars, len(line))
            num_chars += get_chars
            lines.append(line[:get_chars])
            if num_chars == min_len:
                yield l, lines
                break


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_prefix', required=True)
    parser.add_argument('--corpus_path', required=True)
    parser.add_argument('--min_len', type=int, default=200)

    args = parser.parse_args()

    print("Loading data...")
    sys.path.append('../')
    from src.utils import load_letters, split
    bpath = os.path.expanduser(args.corpus_path)
    letters = load_letters(bpath=bpath, subset='')

    print("Loading model...")
    model_path = args.model_prefix + '.pt'
    d_path = args.model_prefix + '.dict.pt'
    model = LMContainer.from_disk(model_path, d_path)
    model.cpu()
    attributor = Attributor(model)

    print("Predicting...")
    labels, texts = zip(*[(l.author[0], lines)
                          for l, lines in crop_letters(
                                  letters, args.min_len)])
    preds = attributor.predict(texts)
    trues = [attributor.authors[idx] for idx in labels]
    print(accuracy_score(preds, trues))

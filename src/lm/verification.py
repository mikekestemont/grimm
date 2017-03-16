
import random
import sys
sys.path.append('../')
import os
import numpy as np
from sklearn.metrics import classification_report
from src.utils import letters2lines, load_letters
from modules import LMContainer


class Attributor(object):
    def __init__(self, model_container):
        """
        Constructor

        Parameters:
        ===========
            - model, a language model container for several authors
        """
        self.model_container = model_container
        self.authors = {a: i for i, a in enumerate(self.model_container.heads)}

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
            text_probas.append(
                [self.model_container.predict_proba(text, author)
                 for author in self.model_container.heads])
        probs = np.array(text_probas, dtype=np.float32)
        norm_probs = (probs - probs.mean(0)) / probs.std(0)
        return norm_probs

    def get_label(self, label):
        """
        Returns the integer label associated with a given str author label
        """
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


def filter_letters(letters, min_len):
    for letter in letters:
        n_chars = len([c for l in letter.lines for c in l])
        if n_chars >= min_len:
            yield letter


def prepare_letters(letters, min_len, use_preprocessor=False):
    labels, texts = [], []
    for letter in filter_letters(letters, min_len):
        labels.append(letter.author[0])  # take initial
        if use_preprocessor:
            text = letters2lines([letter])
        else:
            text = letters2lines([letter], preprocessor=None)
        texts.append(text)
    return labels, texts


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_prefix', required=True)
    parser.add_argument('--corpus_path', required=True)
    parser.add_argument('--min_len', type=int, default=200)
    parser.add_argument('--max_letters', type=int, default=-1)
    parser.add_argument('--use_preprocessor', action='store_true')
    parser.add_argument('--seed', type=int, default=1001)

    args = parser.parse_args()

    print("Loading data...")
    bpath = os.path.expanduser(args.corpus_path)
    letters = load_letters(bpath=bpath, subset='', start_from_line=0)
    random.seed(args.seed)
    random.shuffle(letters)
    labels, texts = prepare_letters(
        letters, args.min_len, use_preprocessor=args.use_preprocessor)
    max_len = len(labels) if args.max_letters < 0 else args.max_letters
    labels, texts = labels[:max_len], texts[:max_len]

    print("Loading model...")
    model_path = args.model_prefix + '.pt'
    d_path = args.model_prefix + '.dict.pt'
    model = LMContainer.from_disk(model_path, d_path).cpu()
    attributor = Attributor(model)

    print("Predicting...")
    probs = attributor.predict_probas(texts)
    preds = probs.argmax(axis=-1)
    trues = [attributor.authors[idx] for idx in labels]
    print(classification_report(trues, preds))

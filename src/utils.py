
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid", {'axes.grid': False})
import numpy as np

from itertools import product
import random
random.seed(1000)
import os
import re
from glob import glob
from collections import namedtuple, Counter
import shutil


def load_letters(bpath='../brothers-grimm-data/',
                 subset='SplittedOCROutputManuscripts/*/',
                 start_from_line=3):
    Letter = namedtuple('letter',
                        ['id1', 'id2', 'author', 'addressee', 'day',
                         'month', 'year', 'words', 'lines', 'fn', 'abspath'])
    letters = []
    for fp in glob(bpath + subset + '*.txt'):
        bn = os.path.basename(fp)
        try:
            id1, id2, send, addr, d, m, y = bn.replace('.txt', '').split('_')
            with open(fp) as f:
                lines = f.readlines()[start_from_line:]
                no_comment = []
                for line in lines:
                    if line.strip().startswith(('Überlieferung',
                                                'Datierung',
                                                'Sachkommentar')):
                        break
                    else:
                        no_comment.append(line)
                words = ' '.join(no_comment).lower().split()
            letter = Letter(
                id1, id2, send, addr, d, m, y, words, no_comment, bn, fp)
            letters.append(letter)
        except:
            print('parsing error:', bn)

    return letters


def load_tales(bpath='../brothers-grimm-data/FairyTales/', year=None):
    Tale = namedtuple('tale', ['title', 'year', 'words'])
    tales = []
    if year:
        for fp in glob(bpath + 'Grimm-'+str(year)+'/*.txt'):
            title = os.path.basename(fp).replace('.txt', '')[5:]\
                        .replace('-', ' ')
            with open(fp) as f:
                words = f.read().lower().split()
            tales.append(Tale(title, year, words))
    return tales


def filter_letters(letters, min_len=500,
                   target_authors={'Jacob-Grimm', 'Wilhelm-Grimm'}):

    letters = [l for l in letters if len(l.words) >= min_len]
    letters = [l for l in letters if l.author in target_authors]

    return letters


def plot_confusion_matrix(cm, target_names,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.tick_params(labelsize=6)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=90)
    plt.yticks(tick_marks, target_names)
    thresh = cm.max() / 2.
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, round(cm[i, j], 2),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def save_letters(letters, target_dir='clean/',
                 use_original_fname=False, normalize_whitespace=True):
    try:
        shutil.rmtree(target_dir)
    except:
        pass
    os.mkdir(target_dir)

    for l in letters:
        if use_original_fname:
            fn = l.fn
        else:
            fn = '-'.join([l.year, l.month, l.day]) + '_' + \
                 l.author + '-'.join([l.id1, l.id2]) + '.txt'
        if normalize_whitespace:
            text = ' '.join(l.words)
        else:
            text = '\n'.join(l.lines)
        with open(target_dir + fn, 'w') as f:
            f.write(text)


def split(letters, test=0.1, dev=0.1):
    jacob, wilhelm = [], []
    pred = {'Jacob-Grimm': 0, 'Wilhelm-Grimm': 1}
    for l in letters:
        (jacob, wilhelm)[pred[l.author]].append(l)
    return jacob, wilhelm


def move_letters(letters, source_dir, target_dir):
    for l in letters:
        shutil.copyfile(l.abspath, os.path.join(target_dir, l.fn))


def make_preprocessor(
        keep_newlines=False,
        normalize_whitespace=True,
        replace_num=True,
        lowercase=True):
    def preprocessor(lines):
        output = []
        for line in lines:
            if len(line.split()) == 0:
                continue
            if normalize_whitespace:
                line = ' '.join(line.split())
            if replace_num:
                line = re.sub(r'[0-9]', '0', line)
            if lowercase:
                line = line.lower()
            if keep_newlines:
                if len(output) == 0:
                    output = [line]
                else:
                    output[0] += line
            else:
                output.append(line)
        return output
    return preprocessor


def letters2lines(letters, preprocessor=make_preprocessor()):
    preprocessor = preprocessor or (lambda lines: lines)
    lines = []
    for letter in letters:
        for line in preprocessor(letter.lines):
            lines.append(line)
    return lines


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', required=True)
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--test', type=float, default=0.1)
    parser.add_argument(
        '--save_all', action='store_true',
        help='save all letters except test set to dataset/all/')
    args = parser.parse_args()

    letters = load_letters(bpath=args.path)
    J, W = split(filter_letters(letters, min_len=5))
    J_split = int(len(J) * (1 - args.test))
    W_split = int(len(W) * (1 - args.test))
    train_dir = os.path.join(args.output_path, 'dataset/train/')
    train_letters = J[:J_split] + W[:W_split]
    save_letters(train_letters, target_dir=train_dir,
                 normalize_whitespace=False, use_original_fname=True)
    test_dir = os.path.join(args.output_path, 'dataset/test/')
    test_letters = J[J_split:] + W[W_split:]
    save_letters(test_letters, target_dir=test_dir,
                 normalize_whitespace=False, use_original_fname=True)
    if args.save_all:
        all_dir = os.path.join(args.output_path, 'dataset/all/')
        all_but_test = [l for l in letters if l not in test_letters]
        save_letters(all_but_test, target_dir=all_dir,
                     use_original_fname=True,
                     normalize_whitespace=False)

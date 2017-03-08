import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid", {'axes.grid' : False})
import numpy as np

from itertools import product
import random
import os
from glob import glob
from collections import namedtuple, Counter
import shutil


def load_letters(bpath='../brothers-grimm-data/', start_from_line=3):
    Letter = namedtuple('letter', ['id1', 'id2', 'author', 'addressee',
                                   'day', 'month', 'year',
                                   'words'])

    letters = []

    for fp in glob(bpath + 'SplittedOCROutputManuscripts/*/*.txt'):

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

            letters.append(Letter(id1, id2, send, addr, d, m, y, words))

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


def save_letters(letters):
    try:
        shutil.rmtree('clean/')
    except:
        pass
    os.mkdir('clean/')

    for l in letters:
        fn = '-'.join([l.year, l.month, l.day]) + '_' + \
             l.author + '-'.join([l.id1, l.id2]) + '.txt'
        text = ' '.join(l.words)
        with open('clean/'+fn, 'w') as f:
            f.write(text)


def split_list(l, test, dev):
    train = 1 - (test + dev)
    assert train + test + dev == 1.0, \
        "Illegal splits [%g, %g, %g]" % (train, test, dev)
    train = int(train * len(l))
    test = int(test * len(l))
    dev = int(dev * len(l))
    return l[:train], l[train:test+train], l[test+train:]


def split(letters, test=0.1, dev=0.1):
    random.shuffle(letters)
    jacob, wilhelm = [], []
    for l in letters:
        (jacob, wilhelm)[l.author == 'Jacob-Grimm'].append(l)
    return split_list(jacob, test, dev), split_list(wilhelm, test, dev)

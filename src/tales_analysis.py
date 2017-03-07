import os
from glob import glob
from collections import namedtuple, Counter, OrderedDict
import shutil
from operator import itemgetter


import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd

from vectorization import Vectorizer

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import LeaveOneOut
from sklearn.metrics import accuracy_score, confusion_matrix

from librosa.segment import agglomerative
from HACluster import VNClusterer, Clusterer

from utils import *

from scipy.stats import mannwhitneyu

def loo(X, labels):
    label_encoder = LabelEncoder()
    int_labels = label_encoder.fit_transform(labels)
    print(int_labels)

    clf = SVC(kernel='linear')#, probability=True)
    nb = X.shape[0]
    loo = LeaveOneOut(nb)

    silver, gold = [], []
    for train, test in loo:
        print('.')
        X_train, X_test = X[train], X[test]
        y_test = [int_labels[i] for i in test]
        y_train = [int_labels[i] for i in train]
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        silver.append(pred[0])
        gold.append(y_test[0])

    info = 'Accuracy after SVC-LOO:' + str(accuracy_score(silver, gold))

    # confusion matrix
    plt.clf()
    T = label_encoder.inverse_transform(gold)
    P = label_encoder.inverse_transform(silver)
    cm = confusion_matrix(T, P, labels=label_encoder.classes_)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    np.set_printoptions(precision=2)
    sns.plt.figure()
    plot_confusion_matrix(cm_normalized, target_names=label_encoder.classes_)
    sns.plt.title(info)
    sns.plt.savefig('../figures/conf_matrix_cross_vocab.pdf')

def main():
    try:
        os.mkdir('../figures/')
    except:
        pass

    YEARS = [1812, 1815, 1819, 1837, 1840, 1843, 1850, 1857]

    # load
    letters = filter_letters(load_letters(), min_len=300)
    print('Loaded letters:', len(letters), '->', Counter([l.author for l in letters]))

    for year in YEARS:
        print("-> YEAR:", year)
        tales = load_tales(year=year)
        print('Loaded tales:', len(tales))

        letter_vectorizer = Vectorizer(mfi=1000, ngram_type='word',
                     ngram_size=1, vocabulary=None,
                     vector_space='tf_idf', lowercase=True,
                     min_df=0.0, max_df=1.0, ignore=[])

        letter_X = letter_vectorizer.fit_transform([l.words for l in letters])
        tales_X = letter_vectorizer.transform([t.words for t in tales])
        
        #loo(letter_X, [l.author for l in letters])
        
        letter_label_encoder = LabelEncoder()
        letter_int_labels = letter_label_encoder.fit_transform([l.author for l in letters])

        letter_clf = SVC(kernel='linear', probability=True)
        letter_clf.fit(letter_X, letter_int_labels)

        tale_probas = letter_clf.predict_proba(tales_X)

        print(letter_label_encoder.classes_)
        probas = tale_probas.mean(axis=0)
        print(probas)

        print(tale_probas.shape)

if __name__ == '__main__':
    main()
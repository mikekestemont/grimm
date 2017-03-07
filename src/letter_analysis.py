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


class OrderedCounter(Counter, OrderedDict):
     'Counter that remembers the order elements are first encountered'

     def __repr__(self):
         return '%s(%r)' % (self.__class__.__name__, OrderedDict(self))

     def __reduce__(self):
         return self.__class__, (OrderedDict(self),)



def pca(X, labels, feature_names):

    # from: http://stackoverflow.com/questions/10481990/matplotlib-axis-with-two-scales-shared-origin
    def align_yaxis(ax1, v1, ax2, v2):
        """adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1"""
        _, y1 = ax1.transData.transform((0, v1))
        _, y2 = ax2.transData.transform((0, v2))
        inv = ax2.transData.inverted()
        _, dy = inv.transform((0, 0)) - inv.transform((0, y1 - y2))
        miny, maxy = ax2.get_ylim()
        ax2.set_ylim(miny + dy, maxy + dy)
    def align_xaxis(ax1, v1, ax2, v2):
        """adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1"""
        x1, _ = ax1.transData.transform((v1, 0))
        x2, _ = ax2.transData.transform((v2, 0))
        inv = ax2.transData.inverted()
        dx, _ = inv.transform((0, 0)) - inv.transform((x1 - x2, 0))
        minx, maxx = ax2.get_xlim()
        ax2.set_xlim(minx + dx, maxx + dx)

    prin_comp = PCA(n_components=2)
    pca_matrix = prin_comp.fit_transform(X)
    loadings = prin_comp.components_.transpose()

    plt.clf()
    sns.set_style('dark')
    sns.plt.rcParams['axes.linewidth'] = 0.4

    fig = sns.plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(111)

    x1, x2 = pca_matrix[:,0], pca_matrix[:,1]
    ax1.scatter(x1, x2, 100, edgecolors='none', facecolors='none')

    idxs = {}
    for label in set(labels):
        if label not in idxs:
            idxs[label] = len(idxs)
    
    targets = [idxs[label] for label in labels]

    for x, y, name, cluster_label in zip(x1, x2, labels, targets):
        ax1.text(x, y, name[:3], ha='center', va="center",
                 color=plt.cm.spectral(cluster_label / 10.),
                 fontdict={'family': 'Arial', 'size': 10})

    ax2 = ax1.twinx().twiny()
    l1, l2 = loadings[:,0], loadings[:,1]
    ax2.scatter(l1, l2, 100, edgecolors='none', facecolors='none');
    for x, y, l in zip(l1, l2, feature_names):
        ax2.text(x, y, l ,ha='center', va="center", size=8, color="darkgrey",
            fontdict={'family': 'Arial', 'size': 9})

    var_exp = prin_comp.explained_variance_ratio_
    ax1.set_xlabel('PC1 ('+ str(round(var_exp[0] * 100, 2)) +'%)')
    ax1.set_ylabel('PC2 ('+ str(round(var_exp[1] * 100, 2)) +'%)')

    # align the axes:
    align_xaxis(ax1, 0, ax2, 0)
    align_yaxis(ax1, 0, ax2, 0)
    # add lines through origins:
    plt.axvline(0, ls='dashed', c='lightgrey')
    plt.axhline(0, ls='dashed', c='lightgrey')

    ax2.set_xticklabels([])
    ax2.set_xticks([])
    ax2.set_yticklabels([])
    ax2.set_yticks([])

    sns.plt.savefig('../figures/pca.pdf', bbox_inches=0)

def hca(X, labels, metric='cosine', fontsize=3):
    
    dm = pairwise_distances(X, metric=metric)
    df = pd.DataFrame(data=dm, columns=labels)
    df = df.applymap(lambda x:int(x*1000)).corr()

    # clustermap plotting:
    cm = sns.clustermap(df)
    ax = cm.ax_heatmap

    # xlabels:
    for idx, label in enumerate(ax.get_xticklabels()):
        label.set_rotation('vertical')
        label.set_fontname('Arial')
        label.set_fontsize(fontsize)
    # ylabels:
    for idx, label in enumerate(ax.get_yticklabels()):
        label.set_rotation('horizontal')
        label.set_fontname('Arial')
        label.set_fontsize(fontsize)
    
    cm.savefig('../figures/clustermap.pdf')

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
    sns.plt.savefig('../figures/conf_matrix.pdf')

def oppose(X, labels, feature_names):
    df = pd.DataFrame(X, columns=feature_names)
    df['brothers'] = labels

    for bro in sorted(set(labels)):
        print('-> Comparing brother', bro, 'to other bro')
        A = df[df['brothers'] == bro]
        B = df[df['brothers'] != bro]

        results = []
        for idx, word in enumerate(feature_names):
            stat, pval = mannwhitneyu(A[word], B[word])
            results.append((word, stat, pval))

        results.sort(key=itemgetter(1))

        plt.clf()

        # plot single best:
        for i in range(5):
            fig, ax = plt.subplots(figsize=(6, 10))
            w = results[i][0]
            ax.set_title(w)
            ax.boxplot([A[w], B[w]])
            ax.set_xticklabels([bro, 'rest'])
            plt.tight_layout()
            plt.savefig('../figures/' + bro + '_boxplot_'+str(i+1)+'.pdf')

        # plot top:
        words, scores = [], []
        for w, U, _ in results[:25]:
            words.append(w)
            scores.append(U)

        plt.clf()
        y_pos = np.arange(len(words))
        plt.barh(y_pos, scores[::-1], color="grey", height=0.8, align="center")
        plt.yticks(y_pos, words)
        plt.title('Bro ' + bro)
        plt.xlabel('Mann-Whitney U')
        plt.ylabel('n-gram')
        plt.tight_layout()
        plt.savefig('../figures/' + bro + '_topfeats.pdf')

def vnc(X, labels):
    dist_matrix = pairwise_distances(X, metric='euclidean')
    clusterer = VNClusterer(dist_matrix, linkage='ward')
    clusterer.cluster(verbose=0)
    short_names = [str(y) for y in labels]
    t = clusterer.dendrogram.ete_tree(short_names)
    t.write(outfile='../figures/vnc_clustering.newick')

def segment_cluster(slice_matrix, slice_names, nb_segments):
    slice_matrix = StandardScaler().fit_transform(slice_matrix)
    slice_matrix = np.asarray(slice_matrix).transpose() # librosa assumes that data[1] = time axis
    segment_starts = agglomerative(data=slice_matrix, k=nb_segments)
    break_points = []
    for i in segment_starts:
        if i > 0: # skip first one, since it's always a segm start!
            break_points.append(slice_names[i])
    return break_points

def bootstrap_segmentation(n_iter, nb_mfw_sampled, corpus_matrix,
                           slice_names, nb_segments=3, random_state=2015):
    np.random.seed(random_state)

    corpus_matrix = np.asarray(corpus_matrix)
    sample_cnts = OrderedCounter()
    for sn in slice_names:
        sample_cnts[sn] = []
        for i in range(nb_segments):
            sample_cnts[sn].append(0)

    for nb in range(n_iter):
        print('===============\niteration:', nb+1)
        # sample a subset of the features in our matrix:
        rnd_indices = np.random.randint(low=0, high=corpus_matrix.shape[1], size=nb_mfw_sampled)
        sampled_matrix = corpus_matrix[:,rnd_indices]
    
        # get which breaks are selected and adjust the cnts:
        selected_breaks = segment_cluster(sampled_matrix, slice_names, nb_segments=nb_segments)
        for i, break_ in enumerate(selected_breaks):
            sample_cnts[break_][i] += 1

    sns.set_style('white')
    sns.plt.rcParams['axes.linewidth'] = 0.2
    plt.rcParams['font.family'] = 'arial'
    plt.rcParams['font.size'] = 8
    plt.clf()
    plt.figure(figsize=(8,20))

    sample_names, breakpoints_cnts = zip(*sample_cnts.items())
    pos = [i for i, n in enumerate(sample_names)][::-1] # reverse for legibility
    plt.yticks(pos, [n for n in sample_names])

    axes = plt.gca()
    #axes.set_xlim([0,n_iter])
    colors = sns.color_palette('hls', nb_segments)

    for i in range(nb_segments-1):
        cnts = [c[i] for c in breakpoints_cnts]
        plt.barh(pos, cnts, align='center', color=colors[i], linewidth=0, label="Boundary "+str(i+1))

    plt.tick_params(axis='y', which='both', left='off', right='off', labelleft='on')
    plt.tick_params(axis='x', which='both', top='off')
    plt.legend()
    plt.savefig('../figures/bootstrap_segment'+str(nb_segments)+'.pdf')

def vnc(X, labels):
    dm = pairwise_distances(X, metric='cosine')
    clusterer = VNClusterer(dm, linkage='ward')
    clusterer.cluster(verbose=1)
    clusterer.dendrogram.draw(save=True,
                                labels=labels,
                                fontsize=3,
                                title="VNC Analysis (Ward's Linkage)")

def main():
    try:
        os.mkdir('../figures/')
    except:
        pass

    # load
    letters = filter_letters(load_letters(), min_len=300)
    print('Loaded:', len(letters), '->', Counter([l.author for l in letters]))
    #save_letters(letters)
    
    """
    # vectorize
    vectorizer = Vectorizer(mfi=300, ngram_type='word',
                 ngram_size=1, vocabulary=None,
                 vector_space='tf_std', lowercase=True,
                 min_df=0.0, max_df=1.0, ignore=[])

    X = vectorizer.fit_transform([l.words for l in letters]).toarray()
    X = StandardScaler().fit_transform(X)
    print(vectorizer.feature_names)

    pca(X, [l.author for l in letters], vectorizer.feature_names)
    hca(X, [l.author for l in letters])
    loo(X, [l.author for l in letters])
    oppose(X, [l.author for l in letters], vectorizer.feature_names)
    """

    # one author:
    letters = filter_letters(load_letters(), min_len=300, target_authors={'Jacob-Grimm'})
    jacob_letters = [(int(l.year), l.words) for l in letters if l.year.isdigit()]
    jacob_letters.sort()
    years, texts = zip(*jacob_letters)

    vectorizer = Vectorizer(mfi=2000, ngram_type='word',
                 ngram_size=1, vocabulary=None,
                 vector_space='tf_std', lowercase=True,
                 min_df=0.0, max_df=1.0, ignore=[])

    X = vectorizer.fit_transform(texts).toarray()
    X = StandardScaler().fit_transform(X)
    print(vectorizer.feature_names)

    vnc(X, [str(y) for y in years])
    bootstrap_segmentation(n_iter=1000, nb_mfw_sampled=50, corpus_matrix=X,
        slice_names=[str(y) for y in years], nb_segments=2, random_state=2015)
    

    
if __name__ == '__main__':
    main()
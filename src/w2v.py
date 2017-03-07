import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from nltk import sent_tokenize
from sklearn.cluster import AgglomerativeClustering

from utils import *

from collections import Counter
import gensim

# load and preprocess:
tales = load_tales(year=1857)
text = ' '.join([' '.join(t.words) for t in tales])

# define sentences:
sentences = [sent.split() for sent in sent_tokenize(text)]
print(len(sentences))

# run model;
model = gensim.models.Word2Vec(sentences, min_count=1)


# count most frequent words (mfi):
cnt = Counter()
for sent in sentences:
    cnt.update(sent)
mfi = [t for t, _ in cnt.most_common(300)]

# get full matrix for relevant mfi:
X = np.asarray([model[w] for w in mfi \
                    if w in model], dtype='float32')
print(X.shape)

# initial dim reduction via pca:
X_pca = PCA(n_components=2).fit_transform(X)
print(X_pca.shape)

# real dimension reduction with tsne:
tsne = TSNE(n_components=2)
coor = tsne.fit_transform(X_pca) # unsparsify?

nb_clusters = 8
plt.clf()
sns.set_style('dark')
sns.plt.rcParams['axes.linewidth'] = 0.4
fig, ax1 = sns.plt.subplots()  
# first plot slices:
x1, x2 = coor[:,0], coor[:,1]
ax1.scatter(x1, x2, 100, edgecolors='none', facecolors='none')
# clustering on top (add some colouring):
clustering = AgglomerativeClustering(linkage='ward',
                    affinity='euclidean', n_clusters=nb_clusters)
clustering.fit(coor)
# add names:
for x, y, name, cluster_label in zip(x1, x2, mfi, clustering.labels_):
    ax1.text(x, y, name, ha='center', va="center",
             color=plt.cm.spectral(cluster_label / 10.),
             fontdict={'family': 'Arial', 'size': 8})
# control aesthetics:
ax1.set_xlabel('')
ax1.set_ylabel('')
ax1.set_xticklabels([])
ax1.set_xticks([])
ax1.set_yticklabels([])
ax1.set_yticks([])
sns.plt.savefig("embeddings.pdf", bbox_inches=0)











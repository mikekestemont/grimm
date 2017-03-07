import os
from glob import glob
from collections import namedtuple, Counter

from vectorization import Vectorizer
from sklearn.decomposition import PCA


def load_letters(bpath='../brothers-grimm-data/', start_from_line=3):
    Letter = namedtuple('letter', ['author', 'addressee',
                                    'day', 'month', 'year',
                                    'words'])

    letters = []

    for fp in glob(bpath + 'SplittedOCROutputManuscripts/*/*.txt'):

        bn = os.path.basename(fp)

        try:

            _, _, send, addr, d, m, y = bn.replace('.txt', '').split('_')
            
            #y = int(''.join([c for c in y if c.isdigit()]))

            with open(fp) as f:
                lines = [l.strip() for l in f.readlines()][start_from_line:]
                words = ' '.join(lines).split()

            letters.append(Letter(send, addr, d, m, y, words))

        except:
            print('parsing error:', bn)

    return letters

def filter_letters(letters, min_len=500,
                    target_authors={'Jacob-Grimm', 'Wilhelm-Grimm'}):
    
    letters = [l for l in letters if len(l.words) >= min_len]
    letters = [l for l in letters if l.author in target_authors]

    return letters

def pca(X, labels):
    prin_comp = PCA(n_components=2)
    pca_matrix = prin_comp.fit_transform(X.toarray()) # unsparsify
    pca_loadings = prin_comp.components_.transpose()

    


def main():
    # load
    letters = load_letters()
    letters = filter_letters(letters, min_len=500)
    print('Loaded:')
    print(Counter([l.author for l in letters]))

    for l in letters:
        fn = l.author + '_' + '-'.join([l.day, l.month, l.year]) + '.txt'
        text = ' '.join(l.words)
        with open('clean/'+fn, 'w') as f:
            f.write(text)

    # vectorize
    vectorizer = Vectorizer(mfi=150, ngram_type='word',
                 ngram_size=1, vocabulary=None,
                 vector_space='tf_std', lowercase=True,
                 min_df=0.0, max_df=1.0, ignore=[])

    X = vectorizer.fit_transform([l.words for l in letters])
    print(vectorizer.feature_names)

    pca(X, [l.author for l in letters])
    

if __name__ == '__main__':
    main()
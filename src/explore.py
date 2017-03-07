import os
from glob import glob
from collections import namedtuple

def load_letters(bpath='../brothers-grimm-data/'):
    Letter = namedtuple('letter', ['author', 'addressee',
                                    'day', 'month', 'year', 'words'])

    letters = []

    for fp in glob(bpath + 'SplittedOCROutputManuscripts/*/*.txt'):

        bn = os.path.basename(fp)

        try:

            _, _, send, addr, d, m, y = bn.replace('.txt', '').split('_')
            
            #y = int(''.join([c for c in y if c.isdigit()]))

            with open(fp) as f:
                lines = [l.strip() for l in f.readlines()][1:]
                words = ' '.join(lines).split()

            letters.append(Letter(send, addr, d, m, y, words))

        except:
            print('parsing error:', bn)

    return letters

def filter_letters(letters, min_len=50,
                    target_authors={'Jacob-Grimm', 'Wilhelm-Grimm'}):
    
    letters = [l for l in letters if len(l.words) >= min_len]
    letters = [l for l in letters if l.author in target_authors]

    return letters



def main():
    letters = load_letters()
    letters = filter_letters(letters)
    print(len(letters))

if __name__ == '__main__':
    main()
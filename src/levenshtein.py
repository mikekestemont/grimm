

def levenshtein(a, b):
    "Calculates the Levenshtein distance between a and b."
    n, m = len(a), len(b)
    if n > m:
        # Make sure n <= m, to use O(min(n,m)) space
        a, b = b, a
        n, m = m, n
    current = range(n+1)
    for i in range(1, m+1):
        previous, current = current, [i]+[0]*n
        for j in range(1, n+1):
            add, delete = previous[j]+1, current[j-1]+1
            change = previous[j-1]
            if a[j-1] != b[i-1]:
                change = change + 1
            current[j] = min(add, delete, change)
    return current[n]


class Levenshtein(object):
    def __init__(self, *words):
        self.n_words = len(words)
        self.dists = [[None for i in range(self.n_words)]
                      for j in range(self.n_words)]
        self._current_idx = 0
        self._encode = {}
        self._decode = []
        for w in words:
            self._encode_new(w)
        for a_idx in range(self.n_words):
            for b_idx in range(self.n_words):
                if not self.dists[a_idx][b_idx]:
                    a, b = self.decode(a_idx), self.decode(b_idx)
                    self.dists[a_idx][b_idx] = levenshtein(a, b)

    def _encode_new(self, w):
        try:
            idx = self._encode[w]
        except KeyError:
            idx = self._current_idx
            self._encode[w] = idx
            self._decode += [w]
            self._current_idx += 1
        return idx

    def encode(self, w):
        return self._encode[w]

    def decode(self, idx):
        return self._decode[idx]

    def dist(self, a, b):
        a_idx, b_idx = self.encode(a), self.encode(b)
        return self.dists[a_idx][b_idx]

    def dists_to(self, a):
        return [(self.decode(b), d)
                for (b, d) in enumerate(self.dists[self.encode(a)])]

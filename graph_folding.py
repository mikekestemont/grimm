
from collections import defaultdict


def read_lines(path):
    with open(path, 'r+') as f:
        for line in f:
            yield line.strip()


def read_tracer_score(path):
    out = {}
    for line in read_lines(path):
        id1, id2, feat_overlap, sim = line.split('\t')
        out[int(id1), int(id2)] = {'feat_overlap': feat_overlap, 'sim': sim}
    return out


def read_wet(path):
    out = {}
    for line in read_lines(path):
        s_id, score = line.strip('\t')
        out[s_id] = score
    return out


def add_cleanliness(tracer_scores, ocr, htr, ocr_first=True):
    for id1, id2 in tracer_scores:
        try:
            id1_wet = ocr[id1] if ocr_first else htr[id1]
            id2_wet = htr[id2] if ocr_first else ocr[id2]
            tracer_scores[id1, id2]['wet'] = (id1_wet, id2_wet)
        except KeyError:
            print("Warning: couldn't find cleanliness score for pair [%d, %d]" % (id1, id2))


def get_bins(width):
    def frange(start, end=None, inc=None):
        """A range function, that does accept float increments...
        (taken from http://code.activestate.com/recipes/
        66472-frange-a-range-function-with-float-increments/)"""
        if end is None:
            end = start + 0.0
            start = 0.0
        if inc is None:
            inc = 1.0
        L = []
        while 1:
            next = start + len(L) * inc
            if inc > 0 and next >= end:
                break
            elif inc < 0 and next <= end:
                break
            L.append(next)
        return L
    return frange(0 + width, 1 + width, width)


def _binning(tracer_scores, width, index):
    """
    index: 0 for ocr, 1 for htr
    """
    sorted_tracer = sorted(tracer_scores.items(), key=lambda x: x[1]['wet'][index])
    binned, current = defaultdict(list), 0
    for to_bin in get_bins(width):
        while current < len(sorted_tracer) and \
              sorted_tracer[current][1]['wet'][index] <= to_bin:
            binned[to_bin].append(sorted_tracer[current])
            current += 1
    return binned


def binning(tracer_scores, width=0.01):
    """
    Run binning on the tracer scores with added cleanliness scores
    """
    return _binning(tracer_scores, width, 0), \
        _binning(tracer_scores, width, 1)


def merge_bins(binned, merge_fn):
    """
    merge bins in place using merge_fn function which takes a list
    of dicts with keys 'feat_overlap', 'sim', and 'wet'.
    """
    for k in binned:
        binned[k] = merge_fn(binned[k])
    return binned


def print_file(binned_ocr, binned_htr, width, merge_fn=len):
    merged_ocr = merge_bins(binned_ocr, merge_fn)
    merged_htr = merge_bins(binned_htr, merge_fn)
    for to_bin in get_bins(width):
        ocr, htr = merged_ocr.get(to_bin, 0), merged_htr.get(to_bin, 0)
        print("%f\t%f\t%g\t%g" % (to_bin - width, to_bin, ocr, htr))


def generate(tracer_scores, ocr, htr, width):
    add_cleanliness(tracer_scores, ocr, htr)
    binned_ocr, binned_htr = binning(tracer_scores, width=width)
    ocr_count = sum([len(v) for v in binned_ocr.values()])
    htr_count = sum([len(v) for v in binned_htr.values()])
    assert len(tracer_scores) == ocr_count == htr_count, \
        "Illegal binnings [%d, %d should be %d]" % (
            len(tracer_scores), ocr_count, htr_count)
    print_file(binned_ocr, binned_htr, width)


def generate_real(tracer_scores_path, ocr_path, htr_path, width):
    tracer_scores = read_tracer_score(tracer_scores_path)
    ocr, htr = read_wet(ocr_path), read_wet(htr_path)
    generate(tracer_scores, ocr, htr, width)


class DummyErrorGenerator(object):
    def __init__(self, itemgenerator):
        self.out = {}
        self.itemgenerator = itemgenerator

    def __getitem__(self, idx):
        try:
            return self.out[idx]
        except KeyError:
            out = self.itemgenerator()
            self.out[idx] = out
            return out


def make_generator():
    from random import uniform
    return DummyErrorGenerator(lambda: uniform(0, 1.0))


def generate_dummy(tracer_scores_path, width):
    tracer_scores = read_tracer_score(tracer_scores_path)
    generate(tracer_scores, make_generator(), make_generator(), width)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tracer_scores')
    parser.add_argument('--ocr_path', required=False)
    parser.add_argument('--htr_path', required=False)
    parser.add_argument('--width', type=float, default=0.01)
    parser.add_argument('--generate_dummy', action='store_true')

    args = parser.parse_args()

    if args.generate_dummy:
        generate_dummy(args.tracer_scores, args.width)
    else:
        generate_real(args.tracer_scores,
                      args.ocr_path,
                      args.htr_path,
                      args.width)

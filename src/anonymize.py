
from explore import load_letters
from levenshtein import Levenshtein
from names import wilhelm, jakob


def has_digit(w):
    return any(c.isdigit() for c in w)


def headings(letters, nwords=10):
    for letter in letters:
        for word in letter.words[:nwords]:
            if not word[0].lower() in ('wj'):
                continue
            if has_digit(word):
                continue
            if len(word) < 5:
                continue
            yield word


def anonymize(words, token='<NAME>'):
    return [token if w in wilhelm or w in jakob else w for w in words]


def anonymize_letter(letter, token='<NAME>'):
    letter.words = anonymize(letter.words, token=token)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    args = parser.parse_args()

    letters = load_letters(bpath=args.path)
    words = set(headings(letters))
    dists = Levenshtein(*words)
    print("Wilhelm:\n")
    for w, _ in sorted(dists.dists_to('Wilhelm'), key=lambda x: x[1]):
        print("\t%s" % w)
    print()
    print("Jakob:\n")
    for w, _ in sorted(dists.dists_to('Jakob'), key=lambda x: x[1]):
        print("\t%s" % w)

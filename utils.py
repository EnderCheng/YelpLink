import itertools


def permutation_bigram(array):
    return list(itertools.permutations(array, 2))


def permutation_trigram(array):
    return list(itertools.permutations(array, 3))


if __name__ == "__main__":
    tag_set = ['ADJ', 'ADP', 'ADV', 'CONJ', 'DET', 'NOUN', 'NUM', 'PRT', 'PRON', 'VERB', '.', 'X']
    print permutation_trigram(tag_set)

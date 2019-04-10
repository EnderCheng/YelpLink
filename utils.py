import itertools
import geopy.distance


def permutation_bigram(array):
    return list(itertools.permutations(array, 2))


def permutation_trigram(array):
    return list(itertools.permutations(array, 3))


def calc_distance(lat_1, lon_1, lat_2, lon_2):
    coord_1 = (lat_1, lon_1)
    coord_2 = (lat_2, lon_2)
    return geopy.distance.distance(coord_1, coord_2).km


if __name__ == "__main__":
    # tag_set = ['ADJ', 'ADP', 'ADV', 'CONJ', 'DET', 'NOUN', 'NUM', 'PRT', 'PRON', 'VERB', '.', 'X']
    # print permutation_trigram(tag_set)
    print calc_distance(52.2296756, 21.0122287, 52.406374, 16.9251681)
import itertools
import geopy.distance
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import euclidean


# def kld(p, q):
#     p, q = zip(*filter(lambda (x, y): x != 0 or y != 0, zip(p, q)))
#     p = p + np.spacing(1)
#     q = q + np.spacing(1)
#     return sum([_p * log(_p / _q, 2) for (_p, _q) in zip(p, q)])
#
#
# def jsd_core(p, q):
#     p, q = zip(*filter(lambda (x, y): x != 0 or y != 0, zip(p, q)))
#     M = [0.5 * (_p + _q) for _p, _q in zip(p, q)]
#     p = p + np.spacing(1)
#     q = q + np.spacing(1)
#     M = M + np.spacing(1)
#     return 0.5 * kld(p, M) + 0.5 * kld(q, M)

def all_combination(length):
    array = range(length)
    return list(itertools.combinations(array, 2))


def permutation_bigram(array):
    return list(itertools.permutations(array, 2))


def permutation_trigram(array):
    return list(itertools.permutations(array, 3))


def calc_distance(lat_1, lon_1, lat_2, lon_2):
    coord_1 = (lat_1, lon_1)
    coord_2 = (lat_2, lon_2)
    return geopy.distance.distance(coord_1, coord_2).km


def calc_avg_dist_similarity_score(user_features_prop, user_features_whole_prop):
    jsd_score = 0
    w_score = 0
    for prop in user_features_prop:
        jsd_score += jensenshannon(prop, user_features_whole_prop, 2.0)
        w_score += wasserstein_distance(prop, user_features_whole_prop)
    review_num = len(user_features_prop)
    avg_jsd_score = jsd_score / review_num
    avg_w_score = w_score / review_num
    return avg_jsd_score, avg_w_score


def cal_similarity(all_users_features_whole_prop):
    user_word_scores = []
    user_bigram_scores = []
    user_trigram_scores = []
    for one_user in all_users_features_whole_prop:
        agg_word_jsd_score = 0
        agg_bigram_jsd_score = 0
        agg_trigram_jsd_score = 0
        for other_user in all_users_features_whole_prop:
            if one_user != other_user:
                word_prop = all_users_features_whole_prop[one_user][one_user]['word']
                bigram_prop = all_users_features_whole_prop[one_user][one_user]['bigram']
                trigram_prop = all_users_features_whole_prop[one_user][one_user]['trigram']

                other_word_prop = all_users_features_whole_prop[other_user][one_user]['word']
                other_bigram_prop = all_users_features_whole_prop[other_user][one_user]['bigram']
                other_trigram_prop = all_users_features_whole_prop[other_user][one_user]['trigram']

                word_jsd_score = jensenshannon(word_prop, other_word_prop, 2.0)
                bigram_jsd_score = jensenshannon(bigram_prop, other_bigram_prop, 2.0)
                trigram_jsd_score = jensenshannon(trigram_prop, other_trigram_prop, 2.0)

                word_prop = all_users_features_whole_prop[other_user][other_user]['word']
                bigram_prop = all_users_features_whole_prop[other_user][other_user]['bigram']
                trigram_prop = all_users_features_whole_prop[other_user][other_user]['trigram']

                other_word_prop = all_users_features_whole_prop[one_user][other_user]['word']
                other_bigram_prop = all_users_features_whole_prop[one_user][other_user]['bigram']
                other_trigram_prop = all_users_features_whole_prop[one_user][other_user]['trigram']

                word_jsd_score += jensenshannon(word_prop, other_word_prop, 2.0)
                bigram_jsd_score += jensenshannon(bigram_prop, other_bigram_prop, 2.0)
                trigram_jsd_score += jensenshannon(trigram_prop, other_trigram_prop, 2.0)

                word_jsd_score = word_jsd_score / 2.00
                bigram_jsd_score = bigram_jsd_score / 2.00
                trigram_jsd_score = trigram_jsd_score / 2.00

                agg_word_jsd_score = agg_word_jsd_score + word_jsd_score
                agg_bigram_jsd_score = agg_bigram_jsd_score + bigram_jsd_score
                agg_trigram_jsd_score = agg_trigram_jsd_score + trigram_jsd_score

        agg_word_jsd_score = agg_word_jsd_score / (len(all_users_features_whole_prop) - 1)
        agg_bigram_jsd_score = agg_bigram_jsd_score / (len(all_users_features_whole_prop) - 1)
        agg_trigram_jsd_score = agg_trigram_jsd_score / (len(all_users_features_whole_prop) - 1)

        user_word_scores.append(agg_word_jsd_score)
        user_bigram_scores.append(agg_bigram_jsd_score)
        user_trigram_scores.append(agg_trigram_jsd_score)
    return user_word_scores, user_bigram_scores, user_trigram_scores


def calc_avg_dist_similarity_score_different_users(user_features_prop, all_users_features_whole_prop, index):
    jsd_score = 0
    w_score = 0
    for j in range(0, len(all_users_features_whole_prop)):
        if index != j:
            avg_jsd_score, avg_w_score = calc_avg_dist_similarity_score(user_features_prop,
                                                                        all_users_features_whole_prop[j])
            jsd_score += avg_jsd_score
            w_score += avg_w_score
    final_jsd_score = jsd_score / (len(all_users_features_whole_prop) - 1)
    final_w_score = w_score / (len(all_users_features_whole_prop) - 1)

    return final_jsd_score, final_w_score


def feature_distance(u, v):
    return euclidean(u, v)


# def calc_avg_dist_similarity_score_different_users(user_features, all_features, index):
#     jsd_score = 0
#     w_score = 0
#     for i in range(0, len(all_features)):
#         if i != index:
#             jsd_score += jensenshannon(user_features, all_features[i], 2.0)
#             w_score += wasserstein_distance(user_features, all_features[i])
#     user_number = len(all_features) - 1
#     avg_jsd_score = jsd_score / user_number
#     avg_w_score = w_score / user_number
#     return avg_jsd_score, avg_w_score


if __name__ == "__main__":
    # tag_set = ['ADJ', 'ADP', 'ADV', 'CONJ', 'DET', 'NOUN', 'NUM', 'PRT', 'PRON', 'VERB', '.', 'X']
    # print permutation_trigram(tag_set)
    print(all_combination(5))

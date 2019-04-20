import os
import config
import json
import re
from collections import Counter
from nltk.tokenize import sent_tokenize
import numpy as np
import math
import nltk
import scipy.stats
from utils import permutation_bigram
from utils import permutation_trigram
import pickle
import shutil

nltk.download('punkt')


def read_business_info():
    str_file_path = config.Project_CONFIG['business_file_path']
    b_data = {}
    file_json_data = open(str_file_path, 'r', encoding="utf8")
    for line in enumerate(file_json_data):
        dict_json_data = json.loads(line[1])
        b_info = {}
        b_id = dict_json_data['business_id']
        b_info['city'] = dict_json_data['city']
        b_info['state'] = dict_json_data['state']
        b_info['lat'] = dict_json_data['latitude']
        b_info['lon'] = dict_json_data['longitude']
        b_info['categories'] = dict_json_data['categories']
        b_data[b_id] = b_info
    return b_data


def read_reviews():
    reviews_path = config.Project_CONFIG['user_folder_path']
    files = os.listdir(reviews_path)
    dict_json_data = []
    for user_file in files:
        if not os.path.isdir(user_file):
            file_json_data = open(reviews_path + user_file, 'r', encoding="utf8")
            for line in enumerate(file_json_data):
                temp_json_data = json.loads(line[1])
                dict_json_data.append(temp_json_data)

    return dict_json_data


def print_reviews(review_list):
    for xxx in range(len(review_list)):
        print(review_list[xxx])


def total_number_of_character(text):
    return len(text)


def total_number_of_alpha_character(text):
    alpha_number = sum(c.isalpha() for c in text)
    return alpha_number


def total_number_of_digit_character(text):
    digit_number = sum(c.isdigit() for c in text)
    return digit_number


def total_number_of_upper_case_character(text):
    upper_case_number = sum(1 for c in text if c.isupper())
    return upper_case_number


def total_number_of_space_character(text):
    space_number = sum(c.isspace() for c in text)
    return space_number


def frequency_of_letters(text):
    text = text.lower()
    ret = {}
    alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
                'v', 'w', 'x', 'y', 'z']
    for i in alphabet:
        ret[i] = 0
    for i in text:
        if i in alphabet:
            ret[i] += 1
    return ret.values()


def total_number_of_words(text):
    word_counts = len(re.findall(r'\w+', text))
    return word_counts


def average_word_length(text):
    if total_number_of_words(text) == 0:
        return 0
    avg_word_len = total_number_of_alpha_character(text) / total_number_of_words(text)
    return avg_word_len


def total_number_of_short_words(text):
    short_word_counts = len([word for word in re.findall(r'\w+', text) if len(word) <= 3])
    return short_word_counts


def average_number_of_sentence_character(text):
    tokens = sent_tokenize(text)
    return np.average([len(token) for token in tokens])


def average_number_of_sentence_word(text):
    tokens = sent_tokenize(text)
    return np.average([len(token.split()) for token in tokens])


def number_of_unique_words(text):
    unique_word_counts = Counter(re.findall(r'\w+', text))
    return len(unique_word_counts)


def hapax_legomenon(text):
    words = re.findall(r'\w+', text)
    index = 0
    freq = {key: 0 for key in words}
    for word in words:
        freq[word] += 1
    for word in freq:
        if freq[word] == 1:
            index += 1
    word_num = len(words)
    if word_num == 0:
        return 0
    val = index / word_num
    return val


def hapax_dislegomenon(text):
    words = re.findall(r'\w+', text)
    index = 0
    freq = {key: 0 for key in words}
    for word in words:
        freq[word] += 1
    for word in freq:
        if freq[word] == 2:
            index += 1
    word_num = len(words)
    if word_num == 0:
        return 0
    val = index / word_num
    return val


def honor_r_measure(text):
    words = re.findall(r'\w+', text)
    index = 0
    freq = {key: 0 for key in words}
    for word in words:
        freq[word] += 1
    for word in freq:
        if freq[word] == 1:
            index += 1
    word_number = len(words)
    unique_number = float(len(set(words)))
    if unique_number == 0:
        return 0
    r_value = 100 * math.log(word_number) / max(1, (1 - (index / unique_number)))
    return r_value


def sichel_measure(text):
    words = re.findall(r'\w+', text)
    index = 0
    freq = {key: 0 for key in words}
    for word in words:
        freq[word] += 1
    for word in freq:
        if freq[word] == 2:
            index += 1
    if float(len(set(words))) == 0:
        return 0
    s_value = index / float(len(set(words)))
    return s_value


# def brunets_w_measure(text):
#     words = re.findall(r'\w+', text)
#     a = 0.172
#     unique_word_number = float(len(set(words)))
#     word_number = len(words)
#     if word_number == 1:
#         word_number += 1
#     b_value = (unique_word_number - a) / (math.log10(word_number))
#     return b_value


def yule_k_characteristic(text):
    words = re.findall(r'\w+', text)
    word_number = len(words)
    freq = Counter()
    freq.update(words)
    index = Counter()
    index.update(freq.values())
    m_value = sum([(value * value) * index[value] for key, value in freq.items()])
    if word_number == 0:
        return 0
    k_value = 10000 * (m_value - word_number) / math.pow(word_number, 2)
    return k_value


def shannon_entropy(text):
    words = re.findall(r'\w+', text)
    word_number = len(words)
    freq = Counter()
    freq.update(words)
    temp_array = np.array(list(freq.values()))
    distribution = 1. * temp_array
    distribution /= max(1, word_number)
    sh_value = scipy.stats.entropy(distribution, base=2)
    return sh_value


# def simpson_index(text):
#     words = re.findall(r'\w+', text)
#     freq = Counter()
#     freq.update(words)
#     word_number = len(words)
#     n = sum([1.0 * i * (i - 1) for i in freq.values()])
#     d_value = 1 - (n / (word_number * (word_number - 1)))
#     return d_value


def word_len_freq_dist(text):
    words = re.findall(r'\w+', text)
    ret = {}
    for i in range(1, 21):
        ret[i] = 0
    for word in words:
        if len(word) <= 20:
            if len(word) in ret:
                ret[len(word)] += 1
    return ret.values()


def freq_of_puncuation(text):
    punc = [",", ".", "'", "!", '"', ";", "?", ":", ";"]
    freq = {key: 0 for key in punc}
    for i in text:
        if i in punc:
            freq[i] += 1
    return freq.values()


def freq_of_func_words(text):
    functional_words = """a between in nor some upon
    about both including nothing somebody us
    above but inside of someone used
    after by into off something via
    all can is on such we
    although cos it once than what
    am do its one that whatever
    among down latter onto the when
    an each less opposite their where
    and either like or them whether
    another enough little our these which
    any every lots outside they while
    anybody everybody many over this who
    anyone everyone me own those whoever
    anything everything more past though whom
    are few most per through whose
    around following much plenty till will
    as for must plus to with
    at from my regarding toward within
    be have near same towards without
    because he need several under worth
    before her neither she unless would
    behind him no should unlike yes
    below i nobody since until you
    beside if none so up your
    """

    func_word = functional_words.split()
    freq = {key: 0 for key in func_word}
    words = re.findall(r'\w+', text)
    for word in words:
        if word in func_word:
            freq[word] += 1
    return freq.values()


def pos_tag_frequency(text):
    words = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(words, tagset='universal')
    tag_set = ['ADJ', 'ADP', 'ADV', 'CONJ', 'DET', 'NOUN', 'NUM', 'PRT', 'PRON', 'VERB', '.', 'X']
    tags = [tag[1] for tag in pos_tags]
    return [tags.count(tag) for tag in tag_set]


def pos_tag_bigram_frequency(text):
    ret = {}
    tokens = nltk.word_tokenize(text)
    nltk_bigrams = list(nltk.bigrams(tokens))
    tag_set = ['ADJ', 'ADP', 'ADV', 'CONJ', 'DET', 'NOUN', 'NUM', 'PRT', 'PRON', 'VERB', '.', 'X']
    bigram_tag_set = permutation_bigram(tag_set)

    for value_0, value_1 in bigram_tag_set:
        ret[(value_0, value_1)] = 0

    for bigram_0, bigram_1 in nltk_bigrams:
        type_0 = nltk.pos_tag([bigram_0], tagset='universal')[0][1]
        type_1 = nltk.pos_tag([bigram_1], tagset='universal')[0][1]
        for value_0, value_1 in bigram_tag_set:
            if type_0 == value_0 and type_1 == value_1:
                ret[(value_0, value_1)] += 1
                break
    return ret.values()


def pos_tag_trigram_frequency(text):
    ret = {}
    tokens = nltk.word_tokenize(text)
    nltk_trigrams = list(nltk.trigrams(tokens))
    tag_set = ['ADJ', 'ADP', 'ADV', 'CONJ', 'DET', 'NOUN', 'NUM', 'PRT', 'PRON', 'VERB', '.', 'X']
    trigram_tag_set = permutation_trigram(tag_set)

    for value_0, value_1, value_2 in trigram_tag_set:
        ret[(value_0, value_1, value_2)] = 0

    for trigram_0, trigram_1, trigram_2 in nltk_trigrams:
        type_0 = nltk.pos_tag([trigram_0], tagset='universal')[0][1]
        type_1 = nltk.pos_tag([trigram_1], tagset='universal')[0][1]
        type_2 = nltk.pos_tag([trigram_2], tagset='universal')[0][1]
        for value_0, value_1, value_2 in trigram_tag_set:
            if type_0 == value_0 and type_1 == value_1 and type_2 == value_2:
                ret[(value_0, value_1, value_2)] += 1
                break
    return ret.values()


def number_of_sentence(text):
    number_of_sentences = sent_tokenize(text)
    return len(number_of_sentences)


def number_of_paragraph(text):
    paragraph = 0
    for idx, word in enumerate(text):
        if word == '\n' and not text[idx - 1] == '\n':
            paragraph += 1

    if text[-1] != '\n':  # if the last line is not a new line, count a paragraph.
        paragraph += 1

    return paragraph


def avg_num_of_sentence_of_paragraph(text):
    paragraphs = list(part for part in text.split('\n') if part != '')
    total_sentence = 0.0
    total_paragraph = len(paragraphs)
    for i in range(0, len(paragraphs)):
        paragraph = paragraphs[i]
        number_of_sentences = sent_tokenize(paragraph)
        total_sentence += len(number_of_sentences)
    if total_paragraph == 0:
        return 0
    return total_sentence / total_paragraph


def avg_num_of_character_of_paragraph(text):
    paragraphs = list(part for part in text.split('\n') if part != '')
    total_character = 0.0
    total_paragraph = len(paragraphs)
    for i in range(0, len(paragraphs)):
        paragraph = paragraphs[i]
        number_of_character = len(paragraph)
        total_character += number_of_character
    if total_paragraph == 0:
        return 0
    return total_character / total_paragraph


def avg_num_of_word_of_paragraph(text):
    paragraphs = list(part for part in text.split('\n') if part != '')
    total_word = 0.0
    total_paragraph = len(paragraphs)
    for i in range(0, len(paragraphs)):
        paragraph = paragraphs[i]
        number_of_word = len(re.findall(r'\w+', paragraph))
        total_word += number_of_word
    if total_paragraph == 0:
        return 0
    return total_word / total_paragraph


def text_feature_extract(text):
    character_features_values = [total_number_of_character(text)] + [total_number_of_alpha_character(text)] \
                                + [total_number_of_digit_character(text)] \
                                + [total_number_of_upper_case_character(text)] \
                                + [total_number_of_space_character(text)] + list(frequency_of_letters(text))

    word_features_values = [total_number_of_words(text)] + [average_word_length(text)] + \
                           [total_number_of_short_words(text)] + \
                           [average_number_of_sentence_character(text)] + [average_number_of_sentence_word(text)] + \
                           [number_of_unique_words(text)] + [hapax_legomenon(text)] + [hapax_dislegomenon(text)] + \
                           [honor_r_measure(text)] + [sichel_measure(text)] + \
                           [yule_k_characteristic(text)] + [shannon_entropy(text)] + \
                           list(word_len_freq_dist(text))

    synatic_features_values = list(freq_of_puncuation(text)) + list(freq_of_func_words(text)) + \
                              list(pos_tag_frequency(text)) + \
                              list(pos_tag_bigram_frequency(text)) + list(pos_tag_trigram_frequency(text))

    structure_features_values = [number_of_sentence(text)] + [number_of_paragraph(text)] + \
                                [avg_num_of_sentence_of_paragraph(text)] + [avg_num_of_character_of_paragraph(text)] + \
                                [avg_num_of_word_of_paragraph(text)]

    return character_features_values + word_features_values + synatic_features_values + structure_features_values


def word_extraction(sentence):
    words = re.findall(r'\w+', sentence)
    return [word.lower() for word in words]


def bigram_extraction(sentence):
    words = word_extraction(sentence)
    bigrams = list(nltk.bigrams(words))
    return bigrams


def trigram_extraction(sentence):
    words = word_extraction(sentence)
    trigrams = list(nltk.trigrams(words))
    return trigrams


def word_features(text):
    all_sentences = sent_tokenize(text)
    words = []
    vocab = []
    for sentence in all_sentences:
        w = word_extraction(sentence)
        words.extend(w)
        vocab = list(set(words))
    return vocab


def word_feature_count(vocab_word, text):
    all_sentences = sent_tokenize(text)
    words = []
    bag_vector = {}
    for sentence in all_sentences:
        w = word_extraction(sentence)
        words.extend(w)

    for v in vocab_word:
        bag_vector[v] = 0
    for w in words:
        for i, word in enumerate(vocab_word):
            if word == w:
                bag_vector[word] += 1
    return bag_vector


def bigram_features(text):
    all_sentences = sent_tokenize(text)
    vocab = []
    bigrams = []
    for sentence in all_sentences:
        bg = bigram_extraction(sentence)
        bigrams.extend(bg)
        vocab = list(set(bigrams))
    return vocab


def bigram_feature_count(vocab_bigram, text):
    all_sentences = sent_tokenize(text)
    bigrams = []
    bag_vector = {}
    for sentence in all_sentences:
        bg = bigram_extraction(sentence)
        bigrams.extend(bg)

    for v in vocab_bigram:
        bag_vector[v] = 0
    for bg in bigrams:
        for i, bigram in enumerate(vocab_bigram):
            if bigram == bg:
                bag_vector[bigram] += 1
    return bag_vector


def trigram_features(text):
    all_sentences = sent_tokenize(text)
    vocab = []
    trigrams = []
    for sentence in all_sentences:
        tg = trigram_extraction(sentence)
        trigrams.extend(tg)
        vocab = list(set(trigrams))
    return vocab


def trigram_feature_count(vocab_trigram, text):
    all_sentences = sent_tokenize(text)
    trigrams = []
    bag_vector = {}
    for sentence in all_sentences:
        tg = trigram_extraction(sentence)
        trigrams.extend(tg)

    for v in vocab_trigram:
        bag_vector[v] = 0
    for tg in trigrams:
        for i, trigram in enumerate(vocab_trigram):
            if trigram == tg:
                bag_vector[trigram] += 1
    return bag_vector


def bag_of_word_features_extract_user():
    reviews_path = config.Project_CONFIG['user_folder_path']
    files = os.listdir(reviews_path)
    user_features = {}
    for user_file in files:
        comment_text = ''
        filename = user_file.split('.')[0]
        user_features[filename] = {}
        if not os.path.isdir(user_file):
            file_json_data = open(reviews_path + user_file, 'r', encoding="utf8")
            for line in enumerate(file_json_data):
                temp_json_data = json.loads(line[1])
                comment_text += temp_json_data['text']
            temp_feature_w = word_features(comment_text)
            temp_feature_bigram = bigram_features(comment_text)
            temp_feature_trigram = trigram_features(comment_text)
            user_features[filename]['word'] = temp_feature_w
            user_features[filename]['bigram'] = temp_feature_bigram
            user_features[filename]['trigram'] = temp_feature_trigram
    return user_features


# def bag_of_word_features_extract():
#     reviews_path = config.Project_CONFIG['user_folder_path']
#     feature_path = config.Project_CONFIG['feature_folder_path']
#     if not os.path.exists(config.Project_CONFIG['feature_folder_path']):
#         os.makedirs(config.Project_CONFIG['feature_folder_path'])
#     files = os.listdir(reviews_path)
#     comment_text = ''
#     print('load the reviews')
#     for user_file in files:
#         if not os.path.isdir(user_file):
#             file_json_data = open(reviews_path + user_file, 'r', encoding="utf8")
#             for line in enumerate(file_json_data):
#                 temp_json_data = json.loads(line[1])
#                 comment_text += temp_json_data['text']
#     print('loading finish')
#
#     print('start extracting word features')
#     temp_feature_w = word_features(comment_text)
#
#     print('start extracting bigram features')
#     temp_feature_bg = bigram_features(comment_text)
#
#     print('start extracting unigram features')
#     temp_feature_tg = trigram_features(comment_text)
#
#     feature_w = sorted(temp_feature_w)
#     feature_bg = sorted(temp_feature_bg)
#     feature_tg = sorted(temp_feature_tg)
#
#     with open(feature_path + 'word_features', 'wb') as f:
#         pickle.dump(feature_w, f)
#
#     with open(feature_path + 'bigram_features', 'wb') as f:
#         pickle.dump(feature_bg, f)
#
#     with open(feature_path + 'trigram_features', 'wb') as f:
#         pickle.dump(feature_tg, f)


# def load_word_features():
#     feature_path = config.Project_CONFIG['feature_folder_path']
#     with open(feature_path + 'word_features', 'rb') as f:
#         feature_w = pickle.load(f)
#         return feature_w
#
#
# def load_bigram_features():
#     feature_path = config.Project_CONFIG['feature_folder_path']
#     with open(feature_path + 'bigram_features', 'rb') as f:
#         feature_bg = pickle.load(f)
#         return feature_bg
#
#
# def load_trigram_features():
#     feature_path = config.Project_CONFIG['feature_folder_path']
#     with open(feature_path + 'trigram_features', 'rb') as f:
#         feature_tg = pickle.load(f)
#         return feature_tg


def feature_proportion_word(text, features):
    feature_pp = []
    feature_count = word_feature_count(features, text)
    total_count = sum(feature_count.values())
    for value in feature_count.values():
        feature_pp.append(value * 1.00 / total_count)
    return feature_pp


def feature_proportion_bigram(text, features):
    feature_pp = []
    feature_count = bigram_feature_count(features, text)
    total_count = sum(feature_count.values())
    if total_count == 0:
        total_count = 1
    for value in feature_count.values():
        feature_pp.append(value * 1.00 / total_count)
    return feature_pp


def feature_proportion_trigram(text, features):
    feature_pp = []
    feature_count = trigram_feature_count(features, text)
    total_count = sum(feature_count.values())
    if total_count == 0:
        total_count = 1
    for value in feature_count.values():
        feature_pp.append(value * 1.00 / total_count)
    return feature_pp


def city_state_category_features():
    b_data = read_business_info()
    reviews_path = config.Project_CONFIG['user_folder_path']
    files = os.listdir(reviews_path)
    city_list = []
    state_list = []
    cate_list = []
    for user_file in files:
        if not os.path.isdir(user_file):
            file_json_data = open(reviews_path + user_file, 'r', encoding="utf8")
            for line in enumerate(file_json_data):
                temp_json_data = json.loads(line[1])
                b_id = temp_json_data['business_id']
                b_info = b_data[b_id]
                if b_info['categories'] is not None:
                    categories = b_info['categories'].split(',')
                    cate_list.extend(categories)
                city_list.append(b_info['city'])
                state_list.append(b_info['state'])
    city_set = sorted(list(set(city_list)))
    state_set = sorted(list(set(state_list)))
    cate_set = sorted(list(set(cate_list)))
    return city_set + state_set + cate_set


def feature_extract_part():
    feature_path = config.Project_CONFIG['feature_folder_path']
    print(feature_path + 'feature_dataset_text_'+config.Project_CONFIG['feature_type'])
    print(feature_path + 'label_dataset_text_'+config.Project_CONFIG['feature_type'])
    reviews_path = config.Project_CONFIG['user_folder_path']
    files = os.listdir(reviews_path)
    label_dataset = []
    user_id = 0
    feature_dataset = []
    for user_file in files:
        if not os.path.isdir(user_file):
            file_json_data = open(reviews_path + user_file, 'r', encoding="utf8")
            for line in enumerate(file_json_data):
                temp_json_data = json.loads(line[1])
                text = temp_json_data['text']
                text_features = text_feature_extract(text)
                all_features = text_features
                label = user_id
                feature_dataset.append(all_features)
                label_dataset.append(label)
        user_id += 1
        print("{} : {}".format("Finished, user id", user_id))

    if not os.path.exists(feature_path):
        os.makedirs(feature_path)

    with open(feature_path + 'feature_dataset_text_'+config.Project_CONFIG['feature_type'], 'wb') as f:
        pickle.dump(feature_dataset, f)

    with open(feature_path + 'label_dataset_text_'+config.Project_CONFIG['feature_type'], 'wb') as f:
        pickle.dump(label_dataset, f)


def feature_extract():
    feature_path = config.Project_CONFIG['feature_folder_path']
    print(feature_path + 'feature_dataset_'+config.Project_CONFIG['feature_type'])
    print(feature_path + 'label_dataset_'+config.Project_CONFIG['feature_type'])
    b_data = read_business_info()
    other_features = city_state_category_features()
    reviews_path = config.Project_CONFIG['user_folder_path']
    files = os.listdir(reviews_path)
    label_dataset = []
    user_id = 0
    feature_dataset = []
    for user_file in files:
        if not os.path.isdir(user_file):
            file_json_data = open(reviews_path + user_file, 'r', encoding="utf8")
            for line in enumerate(file_json_data):
                temp_json_data = json.loads(line[1])
                text = temp_json_data['text']
                stars = temp_json_data['stars']
                text_features = text_feature_extract(text)
                feature_values = [0] * len(other_features)
                b_id = temp_json_data['business_id']
                b_info = b_data[b_id]
                if b_info['categories'] is not None:
                    categories = b_info['categories'].split(',')
                    for category in categories:
                        index = other_features.index(category)
                        feature_values[index] = stars
                city = b_info['city']
                state = b_info['state']
                c_index = other_features.index(city)
                s_index = other_features.index(state)
                feature_values[c_index] = 1
                feature_values[s_index] = 1
                all_features = text_features + feature_values
                label = user_id
                feature_dataset.append(all_features)
                label_dataset.append(label)
        user_id += 1
        print("{} : {}".format("Finished, user id", user_id))

    if not os.path.exists(feature_path):
        os.makedirs(feature_path)

    with open(feature_path + 'feature_dataset_'+config.Project_CONFIG['feature_type'], 'wb') as f:
        pickle.dump(feature_dataset, f)

    with open(feature_path + 'label_dataset_'+config.Project_CONFIG['feature_type'], 'wb') as f:
        pickle.dump(label_dataset, f)


if __name__ == "__main__":
    feature_path_e = config.Project_CONFIG['feature_folder_path']
    if os.path.exists(feature_path_e):
        shutil.rmtree(feature_path_e)
    feature_extract_part()
    # reviews = read_reviews()
    # features = {}
    # all_features = []
    # for review_idx in range(0, len(reviews)):
    #     review_id = reviews[review_idx]['review_id']
    #     review_text = reviews[review_idx]['text']
    #     features[review_id] = feature_extract(review_text)
    #     all_features.append(features[review_id])
    # feature_array = np.array(all_features)
    #
    # x_scalar = StandardScaler()
    # x = x_scalar.fit_transform(feature_array)
    #
    # pca = PCA(n_components=0.99)
    # components = pca.fit_transform(x)
    # kmeans = KMeans(n_clusters=5, n_jobs=-1)
    # kmeans.fit_transform(x)
    # print("labels: ", kmeans.labels_)
    # print_reviews(reviews)

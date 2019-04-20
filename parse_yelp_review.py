import json
import config
import os
import pickle
import random
import statistics
from feature_extraction import total_number_of_words


def aggregate_user_review(review_record):
    user_id = review_record['user_id']
    user_file = open(config.Project_CONFIG['user_folder_path']+user_id+'.json', 'a')
    json.dump(review_record, user_file)
    user_file.write('\n')


def parse_raw_json_file(str_file_path):
    if not os.path.exists(config.Project_CONFIG['user_folder_path']):
        os.makedirs(config.Project_CONFIG['user_folder_path'])
    file_json_data = open(str_file_path, 'r')
    for line in enumerate(file_json_data):
        dict_json_data = json.loads(line[1])
        aggregate_user_review(dict_json_data)


def read_dataset():
    feature_path = config.Project_CONFIG['feature_folder_path']
    with open(feature_path + 'feature_dataset_'+config.Project_CONFIG['feature_type'], 'rb') as f:
        feature_dataset = pickle.load(f)

    with open(feature_path + 'label_dataset_'+config.Project_CONFIG['feature_type'], 'rb') as f:
        label_dataset = pickle.load(f)
    return feature_dataset, label_dataset


def get_feature_dataset(num_t):
    feature_ds, label_ds = read_dataset()
    user_num = max(label_ds) + 1
    user_num_dict = {}
    train_dataset = []
    train_label = []
    train_index = []
    for i in range(0, user_num):
        user_num_dict[i] = num_t
    for i in range(0, len(feature_ds)):
        if user_num_dict[label_ds[i]] > 0:
            train_index.append(i)
            train_dataset.append(feature_ds[i])
            train_label.append(label_ds[i])
            user_num_dict[label_ds[i]] -= 1

    test_dataset = [feature_ds[i] for i in range(0, len(feature_ds)) if i not in train_index]
    test_label = [label_ds[i] for i in range(0, len(label_ds)) if i not in train_index]
    return train_dataset, train_label, test_dataset, test_label


def random_choose_dataset(user_num, min_reviews, max_reviews):
    user_num_list = {}
    reviews_path = config.Project_CONFIG['user_folder_path']
    if not os.path.exists(reviews_path):
        os.makedirs(reviews_path)
    file_json_data = open(config.Project_CONFIG['user_file_path'], 'r', encoding='utf8')
    for line in enumerate(file_json_data):
        dict_json_data = json.loads(line[1])
        user_id = dict_json_data['user_id']
        user_num_list[user_id] = 0
    file_review_data = open(config.Project_CONFIG['review_file_path'], 'r', encoding='utf8')
    for line in enumerate(file_review_data):
        dict_json_data = json.loads(line[1])
        user_num_list[dict_json_data['user_id']] += 1
    print('user number data generation finish')
    user_list = []
    for user_id in user_num_list:
        num = user_num_list[user_id]
        if max_reviews > num > min_reviews:
            user_list.append(user_id)
    if len(user_list) > user_num:
        choose_list = random.sample(user_list, user_num)
    else:
        print('error: not enough users')
        return False
    print('user data generation')
    file_review_data = open(config.Project_CONFIG['review_file_path'], 'r', encoding='utf8')
    for line in enumerate(file_review_data):
        dict_json_data = json.loads(line[1])
        if dict_json_data['user_id'] in choose_list:
            aggregate_user_review(dict_json_data)
    return True


# def number_of_review_records():
#     users = []
#     reviews_path_temp = config.Project_CONFIG['user_folder_path']
#     files_temp = os.listdir(reviews_path_temp)
#     for user_file_temp in files_temp:
#         if not os.path.isdir(user_file_temp):
#             lines = sum(1 for line in open(reviews_path_temp + user_file_temp, 'r', encoding="utf8"))
#         users.append(lines)
#     return users


def dataset_statistic():
    user_num = 0
    user_list = []
    user_num_dict = {}
    file_json_data = open(config.Project_CONFIG['user_file_path'], 'r', encoding='utf8')
    for line in enumerate(file_json_data):
        dict_json_data = json.loads(line[1])
        user_num += 1
        user_list.append(dict_json_data['user_id'])

    for user in user_list:
        user_num_dict[user] = 0

    review_num = 0
    word_count_list = []
    file_review_data = open(config.Project_CONFIG['review_file_path'], 'r', encoding='utf8')
    for line in enumerate(file_review_data):
        dict_json_data = json.loads(line[1])
        review_num += 1
        user_num_dict[dict_json_data['user_id']] += 1
        word_count_list.append(total_number_of_words(dict_json_data['text']))

    print('user number:')
    print(user_num)

    user_num_list = list(user_num_dict.values())
    max_value = max(user_num_list)
    min_value = min(user_num_list)
    avg_value = sum(user_num_list) / len(user_num_list)
    stdv_value = statistics.stdev(user_num_list)

    print('user reviews max, min, avg, stdv:')
    print(max_value)
    print(min_value)
    print(avg_value)
    print(stdv_value)

    print('review number:')
    print(review_num)

    max_value_word = max(word_count_list)
    min_value_word = min(word_count_list)
    avg_value_word = sum(word_count_list) / len(word_count_list)
    stdv_value_word = statistics.stdev(word_count_list)

    print('word count max, min, avg, stdv:')
    print(max_value_word)
    print(min_value_word)
    print(avg_value_word)
    print(stdv_value_word)


if __name__ == "__main__":
    # dataset_statistic()
    random_choose_dataset(100, 100, 110)
    # print(min(number_of_review_records()))
    # parse_raw_json_file(config.Project_CONFIG['review_file_path'])

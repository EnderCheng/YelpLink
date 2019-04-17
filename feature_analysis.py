import re
import config
import json
import os
import datetime
from utils import calc_distance
from utils import calc_avg_dist_similarity_score
from utils import calc_avg_dist_similarity_score_different_users
from utils import cal_similarity
from utils import feature_distance
from feature_extraction import read_reviews
from feature_extraction import feature_extract
from feature_extraction import load_word_features
from feature_extraction import load_bigram_features
from feature_extraction import load_trigram_features
from feature_extraction import feature_proportion_word
from feature_extraction import feature_proportion_bigram
from feature_extraction import feature_proportion_trigram
from feature_extraction import bag_of_word_features_extract
from feature_extraction import bag_of_word_features_extract_user
from sklearn.preprocessing import MinMaxScaler
import numpy as np
# from sklearn.preprocessing import StandardScaler


def average_review_length_per_user(user_number):
    reviews = read_reviews()
    return len(reviews) / user_number


def average_review_word_length():
    reviews = read_reviews()
    all_word_counts = 0
    for review_idx in range(0, len(reviews)):
        review_text = reviews[review_idx]['text']
        word_counts = len(re.findall(r'\w+', review_text))
        all_word_counts += word_counts
    return all_word_counts / len(reviews)


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


def location_analysis():
    reviews_path = config.Project_CONFIG['user_folder_path']
    city_values = []
    state_values = []
    distance_values = []
    std_dist_values = []
    std_dist_city_values = []
    city_distance_values = []
    b_data = read_business_info()
    files = os.listdir(reviews_path)
    for user_file in files:
        if not os.path.isdir(user_file):
            file_json_data = open(reviews_path + user_file, 'r', encoding="utf8")
            city_stat = {}
            state_stat = {}
            points = []
            city_points = []
            for line in enumerate(file_json_data):
                temp_json_data = json.loads(line[1])
                b_id = temp_json_data['business_id']
                b_info = b_data[b_id]
                city = b_info['city']
                state = b_info['state']
                lat = b_info['lat']
                lon = b_info['lon']
                points.append((lat, lon))
                if city in city_stat:
                    city_stat[city] += 1
                else:
                    city_stat[city] = 1
                if state in state_stat:
                    state_stat[state] += 1
                else:
                    state_stat[state] = 1
            max_num = city_stat[max(city_stat, key=city_stat.get)]
            city_num = sum(city_stat.values())
            city_values.append(max_num * 1.00 / city_num)

            file_json_data = open(reviews_path + user_file, 'r', encoding="utf8")
            city_name = max(city_stat, key=city_stat.get)
            for line in enumerate(file_json_data):
                temp_json_data = json.loads(line[1])
                b_id = temp_json_data['business_id']
                b_info = b_data[b_id]
                if b_info['city'] == city_name:
                    lat = b_info['lat']
                    lon = b_info['lon']
                    city_points.append((lat, lon))

            max_num_state = state_stat[max(state_stat, key=state_stat.get)]
            state_num = sum(state_stat.values())
            state_values.append(max_num_state * 1.00 / state_num)

            x = [p[0] for p in points]
            y = [p[1] for p in points]
            centroid = (sum(x) / len(points), sum(y) / len(points))
            distance = 0
            dist_list = []
            for p in points:
                distance += calc_distance(centroid[0], centroid[1], p[0], p[1])
                dist_list.append(calc_distance(centroid[0], centroid[1], p[0], p[1]))
            avg_distance = distance / len(points)
            distance_values.append(avg_distance)
            std_dist_values.append(np.std(dist_list, ddof=1))

            c_x = [c_p[0] for c_p in city_points]
            c_y = [c_p[1] for c_p in city_points]
            c_centroid = (sum(c_x) / len(city_points), sum(c_y) / len(city_points))
            c_distance = 0
            c_dist_list = []
            for c_p in city_points:
                c_distance += calc_distance(c_centroid[0], c_centroid[1], c_p[0], c_p[1])
                c_dist_list.append(calc_distance(c_centroid[0], c_centroid[1], c_p[0], c_p[1]))
            c_avg_distance = c_distance / len(city_points)
            city_distance_values.append(c_avg_distance)
            std_dist_city_values.append(np.std(c_dist_list, ddof=1))

    print(city_values)  # the proportion that the business is in the same city
    print(state_values)  # the proportion that the business is in the same state
    print(distance_values)  # business distance
    print(std_dist_values)  # business distance standard deviation
    print(city_distance_values)  # business distance in the same city
    print(std_dist_city_values)  # business distance in the same city standard deviation


def category_analysis():
    reviews_path = config.Project_CONFIG['user_folder_path']
    cate_values = []
    min_cate_values = []
    max_score_values = []
    std_max_score_values = []
    min_score_values = []
    std_min_score_values = []
    b_data = read_business_info()
    files = os.listdir(reviews_path)
    for user_file in files:
        if not os.path.isdir(user_file):
            cate_stat = {}
            min_categories = []
            avg_score_stat = 0
            avg_min_score_stat = 0
            file_json_data = open(reviews_path + user_file, 'r', encoding="utf8")
            for line in enumerate(file_json_data):
                temp_json_data = json.loads(line[1])
                b_id = temp_json_data['business_id']
                b_info = b_data[b_id]
                if b_info['categories'] is not None:
                    categories = b_info['categories'].split(',')
                    for cate in categories:
                        if cate not in cate_stat:
                            cate_stat[cate] = 1
                        else:
                            cate_stat[cate] += 1

            max_num = cate_stat[max(cate_stat, key=cate_stat.get)]
            cate_num = sum(cate_stat.values())
            cate_values.append(max_num * 1.00 / cate_num)

            for cate in cate_stat:
                if cate_stat[cate] == 1:
                    min_categories.append(cate)
            min_cate_num = len(min_categories)
            min_cate_values.append(min_cate_num * 1.00 / cate_num)

            file_json_data = open(reviews_path + user_file, 'r', encoding="utf8")
            cate_name = max(cate_stat, key=cate_stat.get)
            score_list = []
            min_score_list = []
            for line in enumerate(file_json_data):
                temp_json_data = json.loads(line[1])
                b_id = temp_json_data['business_id']
                b_info = b_data[b_id]
                if b_info['categories'] is not None:
                    categories = b_info['categories'].split(',')
                    score = temp_json_data['stars']
                    if cate_name in categories:
                        avg_score_stat += score
                        score_list.append(score)
                    elif len(list(set(min_categories) & set(categories))) >= 1:
                        avg_min_score_stat += score
                        min_score_list.append(score)
            std_max_score_values.append(np.std(score_list, ddof=1))
            std_min_score_values.append(np.std(min_score_list, ddof=1))
            avg_score_stat = avg_score_stat * 1.00 / max_num
            avg_min_score_stat = avg_min_score_stat * 1.00 / min_cate_num
            max_score_values.append(avg_score_stat)
            min_score_values.append(avg_min_score_stat)
    print(cate_values)  # the proportion that business only appear many times
    print(min_cate_values)  # the proportion that business only appear once
    print(max_score_values)  # score for the business that appear many times
    print(std_max_score_values)
    print(min_score_values)  # score for the business that only appear once
    print(std_min_score_values)


def timestamp_analysis():
    reviews_path = config.Project_CONFIG['user_folder_path']
    date_values = []
    date_div = []
    date_time_format = '%Y-%m-%d %H:%M:%S'
    files = os.listdir(reviews_path)
    for user_file in files:
        if not os.path.isdir(user_file):
            date_stat = []
            file_json_data = open(reviews_path + user_file, 'r', encoding="utf8")
            for line in enumerate(file_json_data):
                temp_json_data = json.loads(line[1])
                date = datetime.datetime.strptime(temp_json_data['date'], date_time_format)
                date_stat.append(date)
            sort_date = sorted(date_stat)
            date_interval = 0
            date_list = []
            for i in range(len(sort_date)):
                if i != len(sort_date) - 1:
                    interval = sort_date[i + 1] - sort_date[i]
                    date_interval += interval.days
                    date_list.append(interval.days)
            date_interval = date_interval * 1.00 / (len(sort_date) - 1)
            date_values.append(date_interval)
            date_div.append(np.std(date_list, ddof=1))
    print(date_values)
    print(date_div)


def stylometry_analysis():
    reviews_path = config.Project_CONFIG['user_folder_path']
    files = os.listdir(reviews_path)
    all_user_features = []
    all_user_features_num = []

    same_user_score = []
    diff_user_score = []
    review_start = 0
    review_end = 0
    print('loading data')
    for user_file in files:
        if not os.path.isdir(user_file):
            file_json_data = open(reviews_path + user_file, 'r', encoding="utf8")
            for line in enumerate(file_json_data):
                temp_json_data = json.loads(line[1])
                features = feature_extract(temp_json_data['text'])
                all_user_features.append(features)
                review_end += 1
            all_user_features_num.append([review_start, review_end])
            review_start = review_end
    print('loading finish')
    print('scaling data')
    mmscaler = MinMaxScaler()
    scalar_features = mmscaler.fit_transform(all_user_features)
    print('scaling finish')
    user_list_features = []
    for i in range(0, len(all_user_features_num)):
        user_features = []
        start = all_user_features_num[i][0]
        end = all_user_features_num[i][1]
        for j in range(start, end):
            user_features.append(scalar_features[j])
        user_list_features.append(user_features)
    user_list_average_features = []
    print('same user distance measuring...')
    for k in range(0, len(user_list_features)):
        user_avg_features = np.array(user_list_features[k]).mean(0)
        user_list_average_features.append(user_avg_features)
        temp_similarity_score = 0
        num = len(user_list_features[k])
        for user_features in user_list_features[k]:
            temp_similarity_score += feature_distance(user_features, user_avg_features)
        similarity_score = temp_similarity_score / num
        same_user_score.append(similarity_score)
    print('diff user distance measuring...')
    for k in range(0, len(user_list_features)):
        agg_similarity_score = 0
        for i in range(0, len(user_list_average_features)):
            similarity_score = 0
            if k != i:
                user_avg_features = user_list_average_features[i]
                num = len(user_list_features[k])
                temp_similarity_score = 0
                for user_features in user_list_features[k]:
                    temp_similarity_score += feature_distance(user_features, user_avg_features)
                similarity_score = temp_similarity_score / num
            agg_similarity_score += similarity_score
        avg_similarity_score = agg_similarity_score / (len(user_list_average_features) - 1)
        diff_user_score.append(avg_similarity_score)

    print(same_user_score)
    print(diff_user_score)


def bag_of_word_analysis_user():
    print('Loading features')
    users_features = bag_of_word_features_extract_user()
    print('Loading Finish')
    reviews_path = config.Project_CONFIG['user_folder_path']
    files = os.listdir(reviews_path)

    word_values_same_user = []
    bigram_values_same_user = []
    trigram_values_same_user = []

    all_users_features_whole_prop = {}
    for user_file in files:
        print('processing ...')
        user_text = ''
        filename = user_file.split('.')[0]
        features_word = users_features[filename]['word']
        features_bigram = users_features[filename]['bigram']
        features_trigram = users_features[filename]['trigram']
        if not os.path.isdir(user_file):
            file_json_data = open(reviews_path + user_file, 'r', encoding="utf8")
            user_features_word_prop = []
            user_features_bigram_prop = []
            user_features_trigram_prop = []
            for line in enumerate(file_json_data):
                temp_json_data = json.loads(line[1])
                user_text += temp_json_data['text']
                feature_word_prop = feature_proportion_word(temp_json_data['text'], features_word)
                feature_bigram_prop = feature_proportion_bigram(temp_json_data['text'], features_bigram)
                feature_trigram_prop = feature_proportion_trigram(temp_json_data['text'], features_trigram)

                user_features_word_prop.append(feature_word_prop)
                user_features_bigram_prop.append(feature_bigram_prop)
                user_features_trigram_prop.append(feature_trigram_prop)
            print('User features proportion...')
            user_features_whole_word_prop = feature_proportion_word(user_text, features_word)
            user_features_whole_bigram_prop = feature_proportion_bigram(user_text, features_bigram)
            user_features_whole_trigram_prop = feature_proportion_trigram(user_text, features_trigram)
            print('User whole features proportion...')

            # distribution similarity between a users' whole distribution and each review's distribution (average)
            avg_score_word = calc_avg_dist_similarity_score(user_features_word_prop,
                                                            user_features_whole_word_prop)
            avg_score_bigram = calc_avg_dist_similarity_score(user_features_bigram_prop,
                                                              user_features_whole_bigram_prop)
            avg_score_trigram = calc_avg_dist_similarity_score(user_features_trigram_prop,
                                                               user_features_whole_trigram_prop)

            word_values_same_user.append(avg_score_word[0])
            bigram_values_same_user.append(avg_score_bigram[0])
            trigram_values_same_user.append(avg_score_trigram[0])

            print('same_user values...')
            # different users
            all_users_features_whole_prop[filename] = {}
            for user_id in users_features:
                all_users_features_whole_prop[filename][user_id] = {}
                tmp_features_word = users_features[user_id]['word']
                tmp_features_bigram = users_features[user_id]['bigram']
                tmp_features_trigram = users_features[user_id]['trigram']

                all_users_features_whole_prop[filename][user_id]['word'] \
                    = feature_proportion_word(user_text, tmp_features_word)
                all_users_features_whole_prop[filename][user_id]['bigram'] \
                    = feature_proportion_bigram(user_text, tmp_features_bigram)
                all_users_features_whole_prop[filename][user_id]['trigram'] \
                    = feature_proportion_trigram(user_text, tmp_features_trigram)
            print('diff_user values...')

    word_values, bigram_values, trigram_values = cal_similarity(all_users_features_whole_prop)

    # print word_values_same_user
    # print bigram_values_same_user
    # print trigram_values_same_user
    # #
    # print word_values
    # print bigram_values
    # print trigram_values

    print([(x + y + z) / 3 for x, y, z in zip(word_values_same_user, bigram_values_same_user,
                                              trigram_values_same_user)])
    print([(x + y + z) / 3 for x, y, z in zip(word_values, bigram_values, trigram_values)])


def bag_of_word_analysis():
    bag_of_word_features_extract()
    reviews_path = config.Project_CONFIG['user_folder_path']
    files = os.listdir(reviews_path)
    # load features
    features_word = load_word_features()
    features_bigram = load_bigram_features()
    features_trigram = load_trigram_features()

    word_values_same_user = []
    bigram_values_same_user = []
    trigram_values_same_user = []

    word_values_same_user_w = []
    bigram_values_same_user_w = []
    trigram_values_same_user_w = []

    word_values = []
    bigram_values = []
    trigram_values = []

    word_values_w = []
    bigram_values_w = []
    trigram_values_w = []

    users_list_features_word_prop = []
    users_list_features_bigram_prop = []
    users_list_features_trigram_prop = []

    all_users_features_whole_word_prop = []
    all_users_features_whole_bigram_prop = []
    all_users_features_whole_trigram_prop = []

    for user_file in files:
        user_text = ''
        if not os.path.isdir(user_file):
            file_json_data = open(reviews_path + user_file, 'r', encoding="utf8")
            user_features_word_prop = []
            user_features_bigram_prop = []
            user_features_trigram_prop = []
            for line in enumerate(file_json_data):
                temp_json_data = json.loads(line[1])
                user_text += temp_json_data['text']
                feature_word_prop = feature_proportion_word(temp_json_data['text'], features_word)
                feature_bigram_prop = feature_proportion_bigram(temp_json_data['text'], features_bigram)
                feature_trigram_prop = feature_proportion_trigram(temp_json_data['text'], features_trigram)

                user_features_word_prop.append(feature_word_prop)
                user_features_bigram_prop.append(feature_bigram_prop)
                user_features_trigram_prop.append(feature_trigram_prop)

            users_list_features_word_prop.append(user_features_word_prop)
            users_list_features_bigram_prop.append(user_features_bigram_prop)
            users_list_features_trigram_prop.append(user_features_trigram_prop)

            user_features_whole_word_prop = feature_proportion_word(user_text, features_word)
            user_features_whole_bigram_prop = feature_proportion_bigram(user_text, features_bigram)
            user_features_whole_trigram_prop = feature_proportion_trigram(user_text, features_trigram)

            # distribution similarity between a users' whole distribution and each review's distribution (average)
            avg_score_word = calc_avg_dist_similarity_score(user_features_word_prop,
                                                            user_features_whole_word_prop)
            avg_score_bigram = calc_avg_dist_similarity_score(user_features_bigram_prop,
                                                              user_features_whole_bigram_prop)
            avg_score_trigram = calc_avg_dist_similarity_score(user_features_trigram_prop,
                                                               user_features_whole_trigram_prop)

            word_values_same_user.append(avg_score_word[0])
            bigram_values_same_user.append(avg_score_bigram[0])
            trigram_values_same_user.append(avg_score_trigram[0])

            word_values_same_user_w.append(avg_score_word[1])
            bigram_values_same_user_w.append(avg_score_bigram[1])
            trigram_values_same_user_w.append(avg_score_trigram[1])

            all_users_features_whole_word_prop.append(user_features_whole_word_prop)
            all_users_features_whole_bigram_prop.append(user_features_whole_bigram_prop)
            all_users_features_whole_trigram_prop.append(user_features_whole_trigram_prop)

    # distribution similarity between one user and all other users (average)
    for i in range(0, len(users_list_features_word_prop)):
        avg_score_word = calc_avg_dist_similarity_score_different_users(users_list_features_word_prop[i],
                                                                        all_users_features_whole_word_prop, i)
        word_values.append(avg_score_word[0])
        word_values_w.append(avg_score_word[1])

    for i in range(0, len(users_list_features_bigram_prop)):
        avg_score_bigram = calc_avg_dist_similarity_score_different_users(users_list_features_bigram_prop[i],
                                                                          all_users_features_whole_bigram_prop, i)
        bigram_values.append(avg_score_bigram[0])
        bigram_values_w.append(avg_score_bigram[1])

    for i in range(0, len(users_list_features_word_prop)):
        avg_score_trigram = calc_avg_dist_similarity_score_different_users(users_list_features_trigram_prop[i],
                                                                           all_users_features_whole_trigram_prop, i)
        trigram_values.append(avg_score_trigram[0])
        trigram_values_w.append(avg_score_trigram[1])

        # word_values.append(avg_score_word[0])
        # bigram_values.append(avg_score_bigram[0])
        # trigram_values.append(avg_score_trigram[0])
        #
        # word_values_w.append(avg_score_word[1])
        # bigram_values_w.append(avg_score_bigram[1])
        # trigram_values_w.append(avg_score_trigram[1])

    # print word_values_same_user
    # print bigram_values_same_user
    # print trigram_values_same_user
    # #
    # print word_values
    # print bigram_values
    # print trigram_values
    print([(x + y + z)/3 for x, y, z in zip(word_values_same_user, bigram_values_same_user, trigram_values_same_user)])
    print([(x + y + z)/3 for x, y, z in zip(word_values, bigram_values, trigram_values)])


if __name__ == "__main__":
    location_analysis()
    category_analysis()
    timestamp_analysis()
    # bag_of_word_analysis()
    bag_of_word_analysis_user()
    stylometry_analysis()

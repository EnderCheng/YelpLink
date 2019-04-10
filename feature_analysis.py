from feature_extraction import read_reviews
import re
import config
import json
import os
from utils import calc_distance


def average_review_length_per_user(user_number):
    reviews = read_reviews()
    return len(reviews)/user_number


def average_review_word_length():
    reviews = read_reviews()
    all_word_counts = 0
    for review_idx in range(0, len(reviews)):
        review_text = reviews[review_idx]['text']
        word_counts = len(re.findall(r'\w+', review_text))
        all_word_counts += word_counts
    return all_word_counts/len(reviews)


def read_business_info():
    str_file_path = config.Project_CONFIG['business_file_path']
    b_data = {}
    file_json_data = open(str_file_path, 'r')
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
    city_distance_values = []
    files = os.listdir(reviews_path)
    b_data = read_business_info()
    for user_file in files:
        if not os.path.isdir(user_file):
            file_json_data = open(reviews_path + user_file, 'r')
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
            city_values.append(max_num*1.00/city_num)

            file_json_data = open(reviews_path + user_file, 'r')
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
            for p in points:
                distance += calc_distance(centroid[0], centroid[1], p[0], p[1])
            avg_distance = distance/len(points)
            distance_values.append(avg_distance)

            c_x = [c_p[0] for c_p in city_points]
            c_y = [c_p[1] for c_p in city_points]
            c_centroid = (sum(c_x) / len(city_points), sum(c_y) / len(city_points))
            c_distance = 0
            for c_p in city_points:
                c_distance += calc_distance(c_centroid[0], c_centroid[1], c_p[0], c_p[1])
            c_avg_distance = c_distance/len(city_points)
            city_distance_values.append(c_avg_distance)

    print city_values
    print state_values
    print distance_values
    print city_distance_values


if __name__ == "__main__":
    # print average_review_length_per_user(1000)
    # print average_review_word_length()
    # print location_of_business('KWywu2tTEPWmR9JnBc0WyQ')
    location_analysis()



import json
import config
import os


def aggregate_user_review(review_record):
    if not os.path.exists(config.Project_CONFIG['user_folder_path']):
        os.makedirs(config.Project_CONFIG['user_folder_path'])
    user_id = review_record['user_id']
    user_file = open(config.Project_CONFIG['user_folder_path']+user_id+'.json', 'a')
    json.dump(review_record, user_file)
    user_file.write('\n')


def parse_raw_json_file(str_file_path):
    file_json_data = open(str_file_path, 'r')
    for line in enumerate(file_json_data):
        dict_json_data = json.loads(line[1])
        aggregate_user_review(dict_json_data)


parse_raw_json_file(config.Project_CONFIG['review_file_path'])

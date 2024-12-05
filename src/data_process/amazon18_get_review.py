import argparse
import collections
import gzip
import html
import json
import os
import random
import re
import torch
from tqdm import tqdm
import numpy as np
from utils import check_path, clean_text, amazon18_dataset2fullname, write_json_file, write_remap_index

def load_review_data(user2id, item2id):

    dataset_full_name = amazon18_dataset2fullname[dataset]
    review_file_path = os.path.join(input_path, 'Review', dataset_full_name + '.json.gz')

    reviews = {}

    with gzip.open(review_file_path, "r") as fp:

        for line in tqdm(fp,desc='Load reviews'):
            inter = json.loads(line)
            try:
                user = inter['reviewerID']
                item = inter['asin']
                if user in user2id and item in item2id:
                    uid = user2id[user]
                    iid = item2id[item]
                else:
                    continue
                if 'reviewText' in inter:
                    review = clean_text(inter['reviewText'])
                else:
                    review = ''
                if 'summary' in inter:
                    summary = clean_text(inter['summary'])
                else:
                    summary = ''
                reviews[str((uid,iid))]={"review":review, "summary":summary}

            except ValueError:
                print(line)

    return reviews

def read_index(file):
    unit2index = {}
    with open(file, 'r') as fp:
        lines = fp.readlines()
        for line in lines:
            unit, index = line.strip().split('\t')
            unit2index[unit] = int(index)
    return unit2index


if __name__ == '__main__':
    dataset = 'Instruments'
    input_path='data/Instruments/meta_dataset'
    output_path ='data/'
    user_2_id_path = os.path.join(output_path, dataset, f'{dataset}.user2id')
    item_2_id_path = os.path.join(output_path, dataset, f'{dataset}.item2id')
    user2index=read_index(user_2_id_path)
    item2index=read_index(item_2_id_path)

    reviews = load_review_data(user2index, item2index)
    write_json_file(reviews, os.path.join(output_path, dataset, f'{dataset}.review.json'))
    


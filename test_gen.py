# -*- coding:utf8 -*-
import codecs
import json
import sys
import nltk
import re
from tqdm import tqdm   
from rouge import Rouge
import numpy as np
def read_metadata(file_to_read):
    passage_list     =  []
    answers_list     =  []
    query_list       =  []
    passage_sent_list=  []
    selected_passage_list =  []
    # description_list =  []
    with open(file_to_read, 'r', encoding='utf8') as data_file:
        for i, line in enumerate(tqdm(data_file)):

            instance = json.loads(line)

            #some answers are blank
            if instance['answers'] == []:
                continue

            passage = ''
            selected_passage = []


            for sentence in instance['passages']:
                if sentence['is_selected'] == 1:
                    selected_passage.append(sentence['passage_text'])
            if selected_passage == []:
                continue

            for i, sentence in enumerate(instance['passages']):
                if i != 0:
                    passage = passage + ' ' + sentence['passage_text']
                else:
                    passage = passage + sentence['passage_text']   

            passage_list.append(passage)


    return len(passage_list)

def write_answers_to_file(path, data):
    with open(path, 'w', encoding='utf8') as data_file:
        data_file.write(json.dumps(data))
if __name__ == '__main__':
    data_size = read_metadata('''/home/zhangs/RC/data/train_v1.1.json''')
    ans_gen_list = []
    for i in range(data_size):
        idx1 = i % 5
        idx12 = 1
        idx13 = 2
        li = [[idx1, idx12],[idx1, idx13]]
        ans_gen_list.append(li)
    write_answers_to_file('''/home/zhangs/RC/data/gen_ans.json''', ans_gen_list)
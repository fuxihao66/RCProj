# -*- coding:utf8 -*-
import codecs
import json
import sys
import nltk
import re
from tqdm import tqdm   
from rouge import Rouge
import numpy as np
import random
def read_metadata(file_to_read, set_type):
    passage_list     =  []
    answers_list     =  []
    query_list       =  []
    passage_sent_list=  []
    selected_passage_list =  []
    query_ids = []
    # description_list =  []
    with open(file_to_read, 'r', encoding='utf8') as data_file:
        for i, line in enumerate(tqdm(data_file)):

            # if len(passage_list) == 500 and set_type == 'train':
            #     break 
            instance = json.loads(line)

            #some answers are blank
            if instance['answers'] == [] and set_type == 'train':
                continue

            passage = ''
            selected_passage = []
            # selected_passage_indics = []
            # passage_to_be_sort = []

            for j, sentence in enumerate(instance['passages']):
                if sentence['is_selected'] == 1:
                    selected_passage.append(sentence['passage_text'].replace("''", '"').replace("``", '"'))
                    # selected_passage_indics.append(j)
            if selected_passage == [] and set_type=='train':
                continue
          

            '''add a temporary part to sort the passage'''
            # for sentence in instance['passages']:
            #     passage_to_be_sort.append(sentence['passage_text'])
            # for j, idx in enumerate(selected_passage_indics):
            #     if j == 0:
            #         passage = passage + passage_to_be_sort[idx]
            #     else:
            #         passage = passage + ' ' + passage_to_be_sort[idx]
            # for idx in range(len(instance['passages'])):
            #     if idx not in selected_passage_indics:
            #         passage = passage + ' ' + passage_to_be_sort[idx]
            for i, sentence in enumerate(instance['passages']):
                if i != 0:
                    passage = passage + ' ' + sentence['passage_text']
                else:
                    passage = passage + sentence['passage_text']   




            passage = passage.replace("''", '"').replace("``", '"')
            passage_list.append(passage)
            selected_passage_list.append(selected_passage)
            
            # answer = ''
            # for j, answer_str in enumerate(instance['answers']):
            #     if j != 0:
            #         answer = answer + ' ' + answer_str
            #     else:
            #         answer = answer + answer_str
            '''in this version, only take the first ans'''
            answer = instance['answers'][0]


            if answer == ' 888-989-4473 ':
                answer = '888-989-4473'
                # print(answer)
            answers_list.append(answer) 

            query_list.append(instance['query'])
            query_ids.append(instance['query_id'])
            # description_list.append(instance['query_type'])

    data_dict = {}
    data_dict['passages']     =  passage_list
    data_dict['answers']      =  answers_list
    data_dict['queries']      =  query_list
    # data_dict['passage_sent'] =  passage_sent_list
    if set_type != 'train':
        data_dict['query_ids']    =  query_ids
    data_dict['passage_selected'] = selected_passage_list
    # data_dict['descriptions'] =  description_list
    return data_dict

    ### this method cannot encode some character very well
    # with open(file_to_write, 'w') as modified_data_file:
    #     modified_data_file.write(json.dumps(data_dict)) 

## signals are also regarded "words"
def word_tokenize(tokens):
    return [token.replace("''", '"').replace("``", '"') for token in nltk.word_tokenize(tokens)]
def Tokenize(para_list):
    
    if isinstance(para_list, list) and isinstance(para_list[0], str):
        l = []
        # split a paragraph by sentences
        sent_tokenize = nltk.sent_tokenize

        for string in para_list:
            li_item = []
            for item in list(map(word_tokenize, sent_tokenize(string))):
                li_item.append(process_tokens(item))
            l.append(li_item)
        return l
    elif isinstance(para_list, str):
        sent_tokenize = nltk.sent_tokenize
        li_item = []
        for item in list(map(word_tokenize, sent_tokenize(para_list))):
            li_item.append(process_tokens(item))
        return li_item
    else:
        raise Exception
def Tokenize_without_sent(para_list):
    if isinstance(para_list, list) and isinstance(para_list[0], str):
        l = []
        for string in para_list:
            l.append(Tokenize_string_word_level(string))
        return l
    elif isinstance(para_list, str):
        return Tokenize_string_word_level(para_list)
    else:
        raise Exception
def Tokenize_string_word_level(para):
    l = process_tokens(word_tokenize(para))
    return l
'''
this method is used to split '/' or '-', 
eg: It's 2017/09/06  or 1997-2017
'''
def process_tokens(temp_tokens):
    tokens = []
    for token in temp_tokens:
        flag = False
        l = ("-", "\u2212", "\u2014", "\u2013", "/", "~", "\u201C", "\u2019", "\u201D", "\u2018", "\u00B0", "\*")
        # \u2013 is en-dash. Used for number to nubmer
        # l = ("-", "\u2212", "\u2014", "\u2013")
        # l = ("\u2013",)
        
        # tokens.extend(re.split("([{}])".format("".join(l)), token))
        temp = re.split("([{}])".format("".join(l)), token)
        for item in temp:
            if len(item) > 1 and item[len(item)-1] == '.':
                tokens.append(item[:len(item)-1])
                tokens.append('.')
            elif len(item) > 0:
                tokens.append(item)
    return tokens

## the case of words should be taken into consideration
def get_word2idx_and_embmat(path_to_file):
    word2vec_dict = {}
    word2idx_dict = {}
    i = 0
    with open(path_to_file, 'r') as vec_file:
        for line in tqdm(vec_file):
            list_of_line = line.split(' ')
            word2vec_dict[list_of_line[0]] = list(map(float, list_of_line[1:]))
            i += 1
            word2idx_dict[list_of_line[0]] = i
    emb_mat = []
    emb_mat.append([0 for i in range(100)])
    for key in word2vec_dict:
        emb_mat.append(word2vec_dict[key])
    emb_mat = np.asarray(emb_mat)
    emb_mat = emb_mat.astype(dtype='float32')
    vacabulary_size = i
    return word2idx_dict, emb_mat, vacabulary_size

def get_char2idx(data_dict):
    char2idx_dict = {}
    i = 0
    for key in data_dict:
        if key == 'passages' or key == 'queries':
            for string in data_dict[key]:
                for char in string:
                    if char not in char2idx_dict:
                        char2idx_dict[char] = i
                        i+=1
    char_vocabulary_size = len(char2idx_dict)
    return char2idx_dict, char_vocabulary_size


def write_to_file( path, data):
        with open(path, 'w', encoding='utf8') as data_file:
            data_file.write(json.dumps(data))  

def get_random_eles_from_list(list_to_select, num_ele):
    return random.sample(list_to_select, num_ele)
def get_flat_idx(wordss, idx):
    return sum(len(words) for words in wordss[:idx[0]]) + idx[1]
def get_phrase(context, wordss, span):

    #span looks like: [ [start_sent_idx, start_word_idx], [end_sent_idx, end_word_idx] ]
    start, stop = span
    #get 1d index in the passage
    flat_start = get_flat_idx(wordss, start)
    flat_stop = get_flat_idx(wordss, stop)
    if flat_start > flat_stop:
        k = flat_start
        flat_start = flat_stop
        flat_stop = k
    
    flat_stop += 1
    words = sum(wordss, [])
    char_idx = 0
    char_start, char_stop = None, None
    for word_idx, word in enumerate(words):
        char_idx = context.find(word, char_idx)
        assert char_idx >= 0
        if word_idx == flat_start:
            char_start = char_idx
        char_idx += len(word)
        if word_idx == flat_stop - 1:
            char_stop = char_idx
    assert char_start is not None
    assert char_stop is not None
    return context[char_start:char_stop]

def get_y_index(y_after_softmax):
    y_indics = []
    for y in y_after_softmax:
        max_value = 0.0
        word_index = 0
        sent_index = 0
        for i, sent in enumerate(y):
            for j, word in enumerate(sent):
                if word > max_value:
                    max_value = word
                    word_index = j
                    sent_index = i
        # print(max_value)
        y_indics.append([sent_index, word_index])
    return y_indics

  
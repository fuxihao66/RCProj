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
        for i, line in enumerate(data_file):

            instance = json.loads(line)

            #some answers are blank
            if instance['answers'] == []:
                continue

            passage = ''
            selected_passage = []
            # passage_sent = []
            for sentence in instance['passages']:
                passage = passage + ' ' + sentence['passage_text']
                if sentence['is_selected'] == 1:
                    selected_passage.append(sentence['passage_text'])
                # passage_sent.append(sentence)
            passage_list.append(passage)
            selected_passage_list.append(selected_passage)
            # passage_sent_list.append(passage_sent)
            
            answer = ''
            for answer_str in instance['answers']:
                answer = answer + ' ' + answer_str
            answers_list.append(answer) 
    
            query_list.append(instance['query'])
            # description_list.append(instance['query_type'])

    data_dict = {}
    data_dict['passages']     =  passage_list
    data_dict['answers']      =  answers_list
    data_dict['queries']      =  query_list
    # data_dict['passage_sent'] =  passage_sent_list
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
        l = ("-", "\u2212", "\u2014", "\u2013", "/", "~", "\u201C", "\u2019", "\u201D", "\u2018", "\u00B0")
        # \u2013 is en-dash. Used for number to nubmer
        # l = ("-", "\u2212", "\u2014", "\u2013")
        # l = ("\u2013",)
        tokens.extend(re.split("([{}])".format("".join(l)), token))
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
    vacabulary_size = i
    return word2idx_dict, emb_mat, vacabulary_size
def get_char2idx(data_dict):
    char2idx_dict = {}
    i = 0
    for key in data_dict:
        if key == 'passaes' or key == 'queries':
            for string in data_dict[key]:
                for char in string:
                    if char not in char2idx_dict:
                        char2idx_dict[char] = i
                        i+=1
    char_vocabulary_size = len(char2idx_dict)
    return char2idx_dict, char_vocabulary_size


    

# def read_batch_data(path_to_file):
    
#     data_set = DataSet()
#     return data_set



    
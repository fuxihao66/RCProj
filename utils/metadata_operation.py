# -*- coding:utf8 -*-
import codecs
import json
import sys
import nltk
import re
import tqdm   

def read_data_as_a_passage(file_to_read):
    passage_list     =  []
    answers_list     =  []
    query_list       =  []
    # description_list =  []
    with open(file_to_read, 'r', encoding='utf8') as data_file:
        for line in data_file:
            instance = json.loads(line)

            #some answers are blank
            if instance['answers'] == []:
                continue

            passage = ''
            for sentence in instance['passages']:
                passage = passage + sentence['passage_text']
            passage_list.append(passage)

            answer = ''
            for answer_str in instance['answers']:
                answer = answer + answer_str
            answers_list.append(answer) 
    
            query_list.append(instance['query'])
            # description_list.append(instance['query_type'])

    data_dict = {}
    data_dict['passages']     =  passage_list
    data_dict['answers']      =  answers_list
    data_dict['queries']      =  query_list
    # data_dict['descriptions'] =  description_list
    
    return data_dict

    ### this method cannot encode some character very well
    # with open(file_to_write, 'w') as modified_data_file:
    #     modified_data_file.write(json.dumps(data_dict)) 




## signals are also regarded "words"
def word_tokenize(tokens):
    return [token.replace("''", '"').replace("``", '"') for token in nltk.word_tokenize(tokens)]
def Tokenize(para):
    
    if isinstance(para, list) and isinstance(para[0], str):
        l = []
        # split a paragraph by sentences
        sent_tokenize = nltk.sent_tokenize

        for string in para:
            li_item = []
            for item in list(map(word_tokenize, sent_tokenize(string))):
                li_item.append(process_tokens(item))
            l.append(li_item)
        return l
    
    else:
        raise Exception

'''
this method is used to split '/' or '-', 
eg: It's 2017/09/06  or 1997-2017
'''
def process_tokens(temp_tokens):
    tokens = []
    for token in temp_tokens:
        flag = False
        l = ("-", "\u2212", "\u2014", "\u2013", "/", "~", '"', "'", "\u201C", "\u2019", "\u201D", "\u2018", "\u00B0")
        # \u2013 is en-dash. Used for number to nubmer
        # l = ("-", "\u2212", "\u2014", "\u2013")
        # l = ("\u2013",)
        tokens.extend(re.split("([{}])".format("".join(l)), token))
    return tokens

def get_rougel_score(summary, reference):
    return 

def get_highest_rl_span(tokenized_para, reference):
    
## the case of words should be taken into consideration
def get_word2vec_from_file(path_to_file):
    word2vec_dict = {}
    word2idx_dict = {}
    i = 0
    with open(path_to_file, 'r') as vec_file:
        for line in tqdm(vec_file):
            list_of_line = line.split(' ')
            word2vec_dict[list_of_line[0]] = list(map(float, list_of_line[1:]))
            word2idx_dict[list_of_line[0]] = i
            i = i+1
    return word2vec_dict, word2idx_dict


if __name__ == '__main__':


    
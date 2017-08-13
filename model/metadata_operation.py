# -*- coding:utf8 -*-
import codecs
import json
import sys
import nltk
import re
import tqdm   
from rouge import Rouge
def read_data_as_a_passage(file_to_read):
    passage_list     =  []
    answers_list     =  []
    query_list       =  []
    passage_sent_list=  []
    # description_list =  []
    with open(file_to_read, 'r', encoding='utf8') as data_file:
        for line in data_file:
            instance = json.loads(line)

            #some answers are blank
            if instance['answers'] == []:
                continue

            passage = ''
            passage_sent = []
            for sentence in instance['passages']:
                passage = passage + sentence['passage_text']
                passage_sent.append(sentence)
            passage_list.append(passage)
            passage_sent_list.append(passage_sent)
            

            ############# 答案去符号！！！！
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
    data_dict['passage_sent'] =  passage_sent_list
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


def get_signal_idxs(string):
    pattern = re.compile(r'''[ -/~\u00B0\u2212\u2014\u2013\u201C\u2019\u201D\u2018]''')    
    signal_idx = 0
    list_of_signal_idxs = []
    while signal_idx < len(string):
        m = pattern.search(string[signal_idx+1:])
        if m:
            temp_idx = m.start()
            signal_idx = signal_idx+temp_idx+1
            list_of_signal_idxs.append(signal_idx)
        else:
            break
    return list_of_signal_idxs
def get_rougel_score(summary, reference, score_type):
    rouge = Rouge()
    scores = rouge.get_scores(reference, summary)
    return scores[0]['rouge-l'][score_type]

def get_idx_sublist(li, subli):
    for idx_li in range(len(li)):
        flag = 1
        for idx_subli in range(len(subli)):
            if subli[idx_subli] != li[idx_li+idx_subli]:
                flag = 0
                break
        if flag == 1:
            return idx_li, idx_li+len(subli)-1
        else:
            continue
    return -1, -1

def trans_idx_1dto2d(idx_start, idx_stop, list2d):
    start_flag = -1
    end_flag = -1

    for i, ele in enumerate(list2d):
        for j, item in enumerate(ele):
            start_flag += 1
            end_flag += 1
            if start_flag == idx_start:
                start_idxs_2d = [i, j]
            if end_flag == idx_stop:
                end_idxs_2d = [i, j]
    return [start_idxs_2d, end_idxs_2d]
def get_highest_rl_span(para, reference, score_type):

    max_rouge = 0
    signal_idxs = get_signal_idxs(para)
    start_idxs = [0]
    for item in signal_idxs:
        start_idxs.append(item+1)
    end_idxs = signal_idxs
    end_idxs.append(len(para))
    for index_start in start_idxs:
        for index_stop in end_idxs:
            if index_start < index_stop:
                temp_score = get_rougel_score(para[index_start: index_stop], reference, score_type)
                if max_rouge < temp_score:
                    best_span_start = index_start
                    best_span_end   = index_stop
                    max_rouge = temp_score
    substring = Tokenize_string_word_level(para[best_span_start: best_span_end]) 
    word_token_para = Tokenize_string_word_level(para)
    sent_token_para = Tokenize(para)
    index_start, index_stop = get_idx_sublist(word_token_para, substring)

    return trans_idx_1dto2d(index_start, index_stop, sent_token_para)

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


    
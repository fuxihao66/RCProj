# -*- coding:utf8 -*-
import nltk
import re
import numpy as np
from rouge import Rouge
import time
from tqdm import tqdm
# def process_tokens(temp_tokens):
#     tokens = []
#     for token in temp_tokens:
#         flag = False
#         l = ("-", "\u2212", "\u2014", "\u2013", "~",  "\u201C", "\u2019", "\u201D", "\u2018", "\u00B0")
#         # \u2013 is en-dash. Used for number to nubmer
#         # l = ("-", "\u2212", "\u2014", "\u2013")
#         # l = ("\u2013",)
#         tokens.extend(re.split("([{}])".format("".join(l)), token))
#     return tokens
# def word_tokenize(tokens):
#     return [token.replace("''", '"').replace("``", '"') for token in nltk.word_tokenize(tokens)]


# def Tokenize(para):
    
#     if isinstance(para, list) and isinstance(para[0], str):
#         l = []
#         # split a paragraph by sentences
#         sent_tokenize = nltk.sent_tokenize

#         for string in para:
#             li_item = []
#             for item in list(map(word_tokenize, sent_tokenize(string))):
#                 li_item.append(process_tokens(item))
#             l.append(li_item)
#         return l
def Tokenize_string_sent_level(para):
    li_item = []
    sent_tokenize = nltk.sent_tokenize
    for item in list(map(word_tokenize, sent_tokenize(para))):
        li_item.append(process_tokens(item))
    return li_item
def Tokenize_string_sent_level_without_process(para):
    sent_tokenize = nltk.sent_tokenize
    return list(map(word_tokenize, sent_tokenize(para)))
def Tokenize_string_word_level(para):

    l = process_tokens(word_tokenize(para))
    return l



# def get_rougel_score(summary, reference, score_type):
#     rouge = Rouge()
#     scores = rouge.get_scores(reference, summary)
#     return scores[0]['rouge-l'][score_type]


# def get_idx_sublist(li, subli):
#     for idx_li in range(len(li)):
#         flag = 1
#         for idx_subli in range(len(subli)):
#             if subli[idx_subli] != li[idx_li+idx_subli]:
#                 flag = 0
#                 break
#         if flag == 1:
#             return idx_li, idx_li+len(subli)-1
#         else:
#             continue
#     return -1, -1

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

# max_gap means the maximun length of gap for answer
'''
1. 答案要做处理： 首字母小写，去标点
2. 对于rouge值为0的要做特殊处理
'''

def get_highest_rl_span(para, reference, max_gap):

    max_rouge = 0
    signal_idxs = get_signal_idxs(para)
    start_idxs = [0]
    for item in signal_idxs:
        start_idxs.append(item+1)
    end_idxs = signal_idxs
    end_idxs.append(len(para))

    for j, index_start in enumerate(start_idxs):
        if max_gap+j > len(end_idxs):
            end_point = len(end_idxs)
        else:
            end_point = max_gap + j
        for index_stop in end_idxs[j: end_point]:
            if index_start < index_stop:
                temp_score = get_rougel_score(para[index_start: index_stop], reference, 'f')
                if max_rouge < temp_score:
                    best_span_start = index_start
                    best_span_end   = index_stop
                    max_rouge = temp_score

    substring = Tokenize_string_word_level(para[best_span_start: best_span_end]) 
    word_token_para = Tokenize_string_word_level(para)
    sent_token_para = Tokenize(para)

    index_start, index_stop = get_idx_sublist(word_token_para, substring)
    print(para[best_span_start: best_span_end])
    print(max_rouge)
    return trans_idx_1dto2d(index_start, index_stop, sent_token_para)


'''
进一步修改  增加 .
'''
def get_signal_idxs(string):
    pattern = re.compile(r'''[ ,()/~\u00B0\u2212\u2014\u2013\u201C\u2019\u201D\u2018-]''')
    signal_idx = 0
    list_of_signal_idxs = []
    while signal_idx < len(string):
        m = pattern.search(string[signal_idx+1:])
        if m:
            temp_idx = m.start()
            signal_idx = signal_idx+temp_idx+1
            
            if string[signal_idx] == ',' and signal_idx < len(string)-1:
                if string[signal_idx+1] == ' ':
                    list_of_signal_idxs.append(signal_idx)
            else:
                list_of_signal_idxs.append(signal_idx)
        else:
            break
    return list_of_signal_idxs

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
            selected_passage_list.append(selected_passage)
            # passage_sent_list.append(passage_sent)
            
            answer = ''
            for i, answer_str in enumerate(instance['answers']):
                if i != 0:
                    answer = answer + ' ' + answer_str
                else:
                    answer = answer + answer_str
            if answer == ' 888-989-4473 ':
                answer = '888-989-4473'
                print(answer)
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
if __name__ == '__main__':
    # print(get_signal_idxs2('''this is 5,500 feet long, you're talling about'''))
    train_data_dict = read_metadata('''/home/zhangs/RC/data/train_v1.1.json''')
    c2i, _ = get_char2idx(train_data_dict)
    print(c2i)
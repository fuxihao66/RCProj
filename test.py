
# -*- coding:utf8 -*-
import codecs
import json
import sys
import nltk
import re
from tqdm import tqdm   
from rouge import Rouge

import numpy as np
def read_metadata(file_to_read, set_type):
    passage_list     =  []
    answers_list     =  []
    query_list       =  []
    passage_sent_list=  []
    selected_passage_list =  []
    # description_list =  []
    with open(file_to_read, 'r', encoding='utf8') as data_file:
        for i, line in enumerate(tqdm(data_file)):

            # if len(passage_list) == 500 and set_type == 'train':
            #     break 
            instance = json.loads(line)

            #some answers are blank
            if instance['answers'] == []:
                continue

            passage = ''
            selected_passage = []
            selected_passage_indics = []
            passage_to_be_sort = []

            for i, sentence in enumerate(instance['passages']):
                if sentence['is_selected'] == 1:
                    selected_passage.append(sentence['passage_text'])
                    selected_passage_indics.append(i)
            if selected_passage == []:
                continue



            '''add a temporary part to sort the passage'''
            for sentence in instance['passages']:
                passage_to_be_sort.append(sentence['passage_text'])
            for i, idx in enumerate(selected_passage_indics):
                if i == 0:
                    passage = passage + passage_to_be_sort[idx]
                else:
                    passage = passage + ' ' + passage_to_be_sort[idx]
            for idx in range(len(instance['passages'])):
                if idx not in selected_passage_indics:
                    passage = passage + ' ' + passage_to_be_sort[idx]
            # for i, sentence in enumerate(instance['passages']):
            #     if i != 0:
            #         passage = passage + ' ' + sentence['passage_text']
            #     else:
            #         passage = passage + sentence['passage_text']   




            passage = passage.replace("''", '"').replace("``", '"')
            passage_list.append(passage)
            selected_passage_list.append(selected_passage)
            
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
    
    return data_dict['passages']
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
def word_tokenize(tokens):
    return [token.replace("''", '"').replace("``", '"') for token in nltk.word_tokenize(tokens)]
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


def get_flat_idx(wordss, idx):
    return sum(len(words) for words in wordss[:idx[0]]) + idx[1]
def get_phrase(context, wordss, span):

    #span looks like: [ [start_sent_idx, start_word_idx], [end_sent_idx, end_word_idx] ]
    start, stop = span
    #get 1d index in the passage
    flat_start = get_flat_idx(wordss, start)
    flat_stop = get_flat_idx(wordss, stop)
    # wordss looks like: [['this', 'is', 'me', '.'],['waht', 'are', 'you', 'doing', '.']]
    words = sum(wordss, [])
    char_idx = 0
    char_start, char_stop = None, None
    for word_idx, word in enumerate(words):
        print(word)
        
        char_idx = context.find(word, char_idx)
        print(char_idx)
        assert char_idx >= 0
        if word_idx == flat_start:
            char_start = char_idx
        char_idx += len(word)
        if word_idx == flat_stop - 1:
            char_stop = char_idx
    assert char_start is not None
    assert char_stop is not None
    return context[char_start:char_stop]
def get_rougel_score_ave(summaries, references, score_type):
    rouge = Rouge()
    scores = rouge.get_scores(references, summaries, avg=True)
    return scores['rouge-l'][score_type]
def get_rougel_score(summary, reference, score_type):
    rouge = Rouge()
    scores = rouge.get_scores(reference, summary)
    return scores[0]['rouge-l'][score_type]
if __name__ == '__main__':
    # summaries = [ ]
    # references = [[]]
    # bleu_score = nltk.translate.bleu_score.sentence_bleu(references, summaries)
    # print(get_rougel_score_ave(summaries, references, 'f'))
    # print(bleu_score)

    reference = 'do you like'
    summary   = 'do you like me.'
    print(get_rougel_score(summary, reference, 'f'))
    # passages = read_metadata('''/home/zhangs/RC/data/train_v1.1.json''', 'train')
    # count = 0
    # for passage in passages:
    #     tokenized_passage = Tokenize_string_word_level(passage)
    #     if len(tokenized_passage) > count:
    #         count = len(tokenized_passage) 
    # print(count)
    # summari = []
    # for summ in summari:
    #     # print(Tokenize_string_word_level(summ))
    #     summ = Tokenize_string_word_level(summ)
    #     print(summ)
    # print(summari)
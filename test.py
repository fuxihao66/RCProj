
# -*- coding:utf8 -*-
import codecs
import json
import sys
import nltk
import re
from tqdm import tqdm   
from rouge import Rouge
import numpy as np
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
if __name__ == '__main__':
    hypothesis = ["open", "the", "file", "what", "are", "you", "doing", "."]
    reference = ["open", "file", "."]
    
    context = ["this is my list", "what are you doing"]
    reference = ['list', "doing"]
    # BLEUscore = nltk.translate.bleu_score.sentence_bleu([reference], hypothesis)
    # print(BLEUscore)
    rouge = Rouge()

    scores = rouge.get_scores(context, reference, avg=True)
    print(scores)
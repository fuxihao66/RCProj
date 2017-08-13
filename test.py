# -*- coding:utf8 -*-
import nltk
import re
import numpy as np
from rouge import Rouge
import time
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
# def Tokenize_string_sent_level(para):
#     li_item = []
#     sent_tokenize = nltk.sent_tokenize
#     for item in list(map(word_tokenize, sent_tokenize(para))):
#         li_item.append(process_tokens(item))
#     return li_item
# def Tokenize_string_sent_level_without_process(para):
#     sent_tokenize = nltk.sent_tokenize
#     return list(map(word_tokenize, sent_tokenize(para)))
# def Tokenize_string_word_level(para):

#     l = process_tokens(word_tokenize(para))

#     return l


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
    print(max_rouge)
    return trans_idx_1dto2d(index_start, index_stop, sent_token_para)
if __name__ == '__main__':
    # string = 'this is not a test-text, but an amazing story! What do you want to do? It''s 2017/06. From 1997-2017'
    # # test = {}
    # # test['char'] = []
    # # ll = Tokenize(string)
    # # for passage in ll:
    # #     cxi = [[list(xijk) for xijk in xij] for xij in passage]
    # #     test['char'].append(cxi)
    # # print(test)
    # li = Tokenize_string_word_level(string)
    # print(enumerate(li))
    # l = {'word':[1.2, 2.2, 3.3], 'key': [2.3, 3.3, 1.6]}
    # emb_mat = np.asarray([l[key] for key in l])
    # string = 'this is my love. he said, "this is my life."'
    # l = Tokenize_string_sent_level_without_process(string)
    # k = Tokenize_string_sent_level(string)
    # print(l)
    # print(k)
    # li = get_word_idx('''this is my l-if~e. 1997/2/3 that is your life''')
    # print(li)
    # li = [1, 3,4,6, 5]
    # sub = [3,4,6]
    # print(get_idx_sublist(li, sub))
    string = '''The average Walgreens salary ranges from approximately $15,000 per year for Customer Service Associate / Cashier to $179,900 per year for District Manager. Average Walgreens hourly pay ranges from approximately $7.35 per hour for Laboratory Technician to $68.90 per hour for Pharmacy Manager. Salary information comes from 7,810 data points collected directly from employees, users, and jobs on Indeed. The average revenue in 2011 of a Starbuck Store was $1,078,000, up  from $1,011,000 in 2010.    The average ticket (total purchase) at domestic Starbuck stores in  No … vember 2007 was reported at $6.36.    In 2008, the average ticket was flat (0.0% change)."}, {"is_selected": 0, "url": "http://news.walgreens.com/fact-sheets/frequently-asked-questions.htm", "passage_text": "In fiscal 2014, Walgreens opened a total of 184 new locations and acquired 84 locations, for a net decrease of 273 after relocations and closings. How big are your stores? The average size for a typical Walgreens is about 14,500 square feet and the sales floor averages about 11,000 square feet. How do we select locations for new stores? There are several factors that Walgreens takes into account, such as major intersections, traffic patterns, demographics and locations near hospitals."}, {"is_selected": 0, "url": "http://www.babson.edu/executive-education/thought-leadership/retailing/Documents/walgreens-strategic-evolution.pdf", "passage_text": "th store in 1984, reaching $4 billion in sales in 1987, and $5 billion two years later. Walgreens ended the 1980s with 1,484 stores, $5.3 billion in revenues and $154 million in profits. However, profit margins remained just below 3 percent of sales, and returns on assets of less than 10 percent. The number of Walgreen stores has risen from 5,000 in 2005 to more than 8,000 at present. The average square footage per store stood at approximately 10,200 and we forecast the figure to remain constant over our review period. Walgreen earned $303 as average front-end revenue per store square foot in 2012. Your Walgreens Store. Select a store from the search results to make it Your Walgreens Store and save time getting what you need. Your Walgreens Store will be the default location for picking up prescriptions, photos, in store orders and finding deals in the Weekly Ad.'''
    reference = '''Approximately $15,000 per year.'''
    start = time.clock()

    print(get_highest_rl_span(string, reference, 'f'))

    elapsed = (time.clock() - start)
    print("Time used:",elapsed)
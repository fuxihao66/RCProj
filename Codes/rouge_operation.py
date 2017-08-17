from rouge import Rouge
import re
from metadata_operation import *

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
        if idx_li+len(subli) > len(li):
            return -1, -1
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
    flag = -1
    for i, ele in enumerate(list2d):
        for j, item in enumerate(ele):
            flag += 1

            if flag == idx_start:
                start_idxs_2d = [i, j]
            if flag == idx_stop:
                end_idxs_2d = [i, j]
    # print(flag)
    print(idx_start)
    print(idx_stop)

    return [start_idxs_2d, end_idxs_2d]
def get_highest_rl_span(para, reference, max_gap):

    max_rouge = 0
    signal_idxs = get_signal_idxs(para)
    start_idxs = [0]
    for item in signal_idxs:
        start_idxs.append(item+1)
    end_idxs = signal_idxs
    end_idxs.append(len(para))

    for j, index_start in enumerate(start_idxs):
        if get_rougel_score(para, reference, 'f') == 0.0:
            return 1, False
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
    # print(para[best_span_start: best_span_end])
    # print(para)
    print(max_rouge)
    # print(substring)
    # print(word_token_para)
    # print(sent_token_para)
    return trans_idx_1dto2d(index_start, index_stop, sent_token_para), True

def get_selected_span(para, selected_span):
    
    substring = Tokenize_string_word_level(selected_span)
    word_token_para = Tokenize_string_word_level(para)
    sent_token_para = Tokenize(para)
    index_start, index_stop = get_idx_sublist(word_token_para, substring)
    
    return trans_idx_1dto2d(index_start, index_stop, sent_token_para)
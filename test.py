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

# max_gap means the maximun length of gap for answer
def get_highest_rl_span(para, reference, max_gap):

    max_rouge = 0
    signal_idxs = get_signal_idxs(para)
    start_idxs = [0]
    for item in signal_idxs:
        start_idxs.append(item+1)
    end_idxs = signal_idxs
    end_idxs.append(len(para))

    start = time.clock()
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
    elapsed = (time.clock() - start)
    print("Time used:",elapsed)

    substring = Tokenize_string_word_level(para[best_span_start: best_span_end]) 
    word_token_para = Tokenize_string_word_level(para)
    sent_token_para = Tokenize(para)

    index_start, index_stop = get_idx_sublist(word_token_para, substring)
    print(para[best_span_start: best_span_end])
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
    string = '''PRESCRIBED FOR: Ginkgo biloba is used for. 1  memory improvement, 2  dementia, 3  Alzheimer's disease, 4  anxiety, 5  multiple sclerosis, 6  tinnitus (ringing in the ears), 7  sexual dysfunction, 8  premenstrual syndrome, 9  dizziness, 10  headache, 11  glaucoma, 12  diabetic eye problems, and. 13  vertigo. DRUG CLASS AND MECHANISM: Ginkgo biloba is a natural herbal supplement. Ginkgo biloba may have antioxidant properties. Ginkgo biloba also slows down platelet binding in the body, which may increase bleeding risks. Ginkgo biloba is commonly used for memory improvement and dementia. It can cause some minor side effects such as stomach upset, headache, dizziness, constipation, forceful heartbeat, and allergic skin reactions. There is some concern that ginkgo leaf extract might increase the risk of liver and thyroid cancers. 1 For vertigo: dosages of 120-160 mg per day of ginkgo leaf extract, divided into two or three doses. 2  For premenstrual syndrome (PMS): 80 mg twice daily, starting on the sixteenth day of the menstrual cycle until the fifth day of the next cycle. Although ginkgo biloba is a natural product, it may still cause side effects. As with any medication or supplement, ginkgo biloba (ginkgo) can cause side effects. Although some people assume that natural products (such as ginkgo biloba) are automatically free of side effects, this is simply not the case. Remember, many poisons and toxins are also natural products. Ginkgo is a prescription herb in Germany. Ginkgo Biloba is especially good when combined with Panax Ginseng. Ginkgo extract has proven benefits to elderly persons. This ancient herb acts to enhance oxygen utilization and thus improves memory, concentration, and other mental faculties. In studies, Ginkgo biloba has been reported as demonstrating anti-oxidant abilities with improvements of the platelet and nerve cell functions and blood flow to the nervous system and brain. It has also been reported as reducing blood viscosity. Other uses for which ginkgo biloba extract is often recommended include depression, diabetes related nerve damage and poor circulation, allergies, vertigo, short-term memory loss, headache, atherosclerosis, tinnitus, cochlear deafness, macular degeneration, diabetic retinopathy, and PMS. In studies, Ginkgo biloba has been reported as demonstrating anti-oxidant abilities with improvements of the platelet and nerve cell functions and blood flow to the nervous system and brain. It has also been reported as reducing blood viscosity. If you suffer from vertigo, the conventional treatment is a drug called meclizine (Antivert, Bonine), which lessens nausea and may also relieve the sensation of spinning, but it doesn't always work and can cause drowsiness, among other side effects. Although ginkgo biloba can be effective in reducing dizziness, it can also cause dizziness as a side effect. Other possible side effects include headache, heart palpitations, gastrointestinal discomfort and skin rash. Consult your doctor before using ginkgo biloba. Ginkgo Biloba for Dizziness. Ginkgo biloba extracts can help alleviate dizziness. Photo Credit Comstock/Comstock/Getty Images. Dizziness -- a feeling that you or your surroundings are spinning -- can be an alarming sensation, but rarely signals a life-threatening condition. Also called vertigo, dizziness can be caused by benign paroxysmal positional vertigo, inner ear inflammations, Meniere's disease, and certain medications. Natural healers often recommend ginkgo biloba to alleviate dizziness. Consult your doctor before taking ginkgo biloba. Ginkgo Biloba for Dizziness. Ginkgo biloba extracts can help alleviate dizziness. Photo Credit Comstock/Comstock/Getty Images. Dizziness -- a feeling that you or your surroundings are spinning -- can be an alarming sensation, but rarely signals a life-threatening condition. Taking ginkgo along with some medications that are change by the liver can increase the effects and side effects of your medication. Before taking ginkgo talk to your healthcare provider if you take any medications that are changed by the liver. 1 For vertigo: dosages of 120-160 mg per day of ginkgo leaf extract, divided into two or three doses. 2  For premenstrual syndrome (PMS): 80 mg twice daily, starting on the sixteenth day of the menstrual cycle until the fifth day of the next cycle.'''
    reference = 'Yes'
    # print(get_highest_rl_span(string, reference, 30))
    print(get_rougel_score(string, reference, 'f'))
    # get_signal_idxs(string)
    # p = Tokenize(string)


    
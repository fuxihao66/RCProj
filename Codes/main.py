import argparse
import json
import math
import os
import shutil
from pprint import pprint
import tensorflow as tf
from tqdm import tqdm
import numpy as np

from model import *
from train import *
from metadata_operation import *
from pre_processing import *
from rouge_operation import *
def main(config):

    
    if config.mode == 'train':
        _train(config)
    # elif config.mode == 'test':
    #     _test(config)
    # elif config.mode == 'forward':
    #     _forward(config)
    else:
        raise ValueError("invalid value for 'mode': {}".format(config.mode))


def get_ridof_blank(train_data_dict):
    passage_for_train = []
    queries_for_train = []
    path_with_blank = '''/home/zhangs/RC/data/Repo_for_transfer/ans_train_text.txt'''
    with open(path_with_blank, 'r') as span_file:
        for i, span in enumerate(span_file):
            if span != '\n':
                para = train_data_dict['passages'][i]
                passage_for_train.append(para)
                queries_for_train.append(train_data_dict['queries'][i])
    return passage_for_train, queries_for_train
def do_get_phrase():
    train_data_dict = read_metadata('''/home/zhangs/RC/data/train_v1.1.json''', 'train')
    train_data_dict_backup = read_metadata('''/home/zhangs/RC/data/train_v1.1.json''', 'train')
    passage_for_train, queries_for_train = get_ridof_blank(train_data_dict)
    train_data_dict['passages'] = passage_for_train
    train_data_dict['queries']  = queries_for_train
    
    path_ans_indics = '''/home/zhangs/RC/data/Repo_for_transfer/ans_train_nonblank_indics.json'''
    path_span_test = '''/home/zhangs/RC/data/Repo_for_transfer/test.txt'''
    list_extracted = []
    with open(path_ans_indics, 'r') as span_file:
        for line in span_file:
            instance = json.loads(line)
            for i,span in enumerate(instance):
                para = train_data_dict['passages'][i]
                try:
                    list_extracted.append(get_phrase(para, Tokenize_without_sent(para), span))
                except:
                    print(i)
                    print(para)
                    print(span)
                    return
    with open(path_span_test, 'w') as ex_file:
        for instance in list_extracted:
            ex_file.write(instance+'\n')
def _train(config):

    train_data_dict = read_metadata('''/home/zhangs/RC/data/train_v1.1.json''', 'train')
    passage_for_train, queries_for_train = get_ridof_blank(train_data_dict)
    train_data_dict['passages'] = passage_for_train
    train_data_dict['queries']  = queries_for_train
    


    '''TODO: the char dict should also contain dev-set'''
    char2idx_dict, char_vocabulary_size = get_char2idx(train_data_dict)
    

    word2idx_dict, emb_mat, vocabulary_size = get_word2idx_and_embmat('''/home/zhangs/RC/data/glove.6B.100d.txt''')
    
    train_data = DataSet(train_data_dict)
    train_data.init_with_ans_file('''/home/zhangs/RC/data/Repo_for_transfer/ans_train_nonblank_indics.json''', config.batch_size, 'train')
    

    config.emb_mat = emb_mat
    config.word_vocab_size = vocabulary_size
    config.char_vocab_size = char_vocabulary_size

    with tf.name_scope("model"):
        model = Model(config, word2idx_dict, char2idx_dict)
    # models = get_multi_models(config, word2idx_dict, char2idx_dict)

    with tf.name_scope("trainer"):
        trainer = single_GPU_trainer(config, model)
    # trainer = MultiGPUTrainer(config, models)

    
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    

    global_step = 0


    train_writer = tf.summary.FileWriter('/home/zhangs/RC/data/FINAL', sess.graph)

    init = tf.global_variables_initializer()
    sess.run(init)

    batch_list = train_data.get_batch_list()
    batch_list_length = len(batch_list)
    batch_num = 5


    for i in range(config.num_epochs):

        for i in range(int(math.ceil(batch_list_length/batch_num))):
        # for i in range(int(math.ceil(batch_list_length/config.num_gpus))):
            sub_batch_list = get_random_eles_from_list(batch_list, batch_num)
            # sub_batch_list = get_random_eles_from_list(batch_list, config.num_gpus)

            # global_step = sess.run(models[0].global_step) + 1
            # print(global_step)
            # if global_step == 10000:
            #     trainer.change_lr(0.3)
            # loss, summary, train_op = trainer.step(sess, sub_batch_list, True)
            # train_writer.add_summary(summary, global_step)
            # print(loss)

            for batch in sub_batch_list:
                global_step = sess.run(model.global_step) + 1  # +1 because all calculations are done after step
                get_summary = True
                print(global_step)

                if global_step == 12000:
                    trainer.change_lr(0.3)

                loss, summary, train_op = trainer.step(sess, batch, get_summary=get_summary)
                train_writer.add_summary(summary, global_step)

                print(loss)
    

    '''start to evaluate via dev-set'''
    dev_data_dict = read_metadata('''/home/zhangs/RC/data/dev_v1.1.json''', 'dev')
    dev_data_dict_backup = read_metadata('''/home/zhangs/RC/data/dev_v1.1.json''', 'dev')
    dev_data   = DataSet(dev_data_dict)
    dev_data.init_without_ans(config.batch_size, 'dev')
    ans_list = dev_data.answers_list
    dev_batches = dev_data.get_batch_list()

    summaries = []
    for j, batch in enumerate(dev_batches):
        feed_dict = model.get_feed_dict(batch, None, False)
        yp, yp2 = sess.run([model.yp, model.yp2], feed_dict=feed_dict)   
        yp = get_y_index(yp)
        yp2= get_y_index(yp2)
        for i in range(len(batch['x'])):
            
            # print(dev_data_dict_backup['passages'][j*config.batch_size+i])
            
            words = batch['x'][i]
            # wo = wordss[yp[i][0]:yp2[i][0]+1]
            # wo[0] = wo[0][yp[i][1]:]
            # wo[len(wo)-1] = wo[len(wo)-1][:yp2[i][1]+1]
            # print(wo)
            try:
                summary = get_phrase(dev_data_dict_backup['passages'][j*config.batch_size+i], words, [yp[i], yp2[i]])
                summaries.append(summary)  
            except:
                print(yp[i])
                print(yp2[i])
                print(dev_data_dict_backup['passages'][j*config.batch_size+i])
                print(words)
    

    path_result = '''/home/zhangs/RC/data/out_train_dev/dev_out.txt'''
    with open(path_result, 'w') as out_file:
        for summary in summaries:
            out_file.write(summary+'\n')

    
    ## TODO apply the model to training data
    train_data_dict = read_metadata('''/home/zhangs/RC/data/train_v1.1.json''', 'dev')
    train_data_dict_backup = read_metadata('''/home/zhangs/RC/data/train_v1.1.json''', 'dev')
    train_data = DataSet(train_data_dict)
    train_data.init_without_ans(config.batch_size, 'dev')
    train_batches = train_data.get_batch_list()

    summaries = []

    for j, batch in enumerate(train_batches):
        feed_dict = model.get_feed_dict(batch, None, False)
        yp, yp2 = sess.run([model.yp, model.yp2], feed_dict=feed_dict)   
        yp = get_y_index(yp)
        yp2= get_y_index(yp2)
        for i in range(len(batch['x'])):
            words = batch['x'][i]
            try:
                summary = get_phrase(train_data_dict_backup['passages'][j*config.batch_size+i], words, [yp[i], yp2[i]])
                summaries.append(summary)  
            except:
                print(yp[i])
                print(yp2[i])
                print(words)

    path_result = '''/home/zhangs/RC/data/out_train_dev/train_out.txt'''
    with open(path_result, 'w') as out_file:
        for summary in summaries:
            out_file.write(summary+'\n')
    # rouge_score = get_rougel_score_ave(summaries, dev_data_dict['answers'], 'f')

    # summ = []
    # for summary in summaries:
    #     summ.append(Tokenize_string_word_level(summary))
    # reference = []
    # for ref in dev_data_dict['answers']:
    #     reference.append([Tokenize_string_word_level(ref)])
    # bleu_score = nltk.translate.bleu_score.corpus_bleu(reference, summ)
    
    # print(rouge_score)
    # print(bleu_score)
            

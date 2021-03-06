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



def _train(config):
    '''combine train and dev to generate char_dict'''
    train_data_dict = read_metadata('''/home/zhangs/RC/data/train_v1.1.json''', 'train')
    dev_data_dict   = read_metadata('''/home/zhangs/RC/data/dev_v1.1.json''', 'dev')
    dev_data_dict['passages'].extend(train_data_dict['passages'])
    dev_data_dict['queries'].extend(train_data_dict['queries'])
    char2idx_dict, char_vocabulary_size = get_char2idx(dev_data_dict)
    

    word2idx_dict, emb_mat, vocabulary_size = get_word2idx_and_embmat('''/home/zhangs/RC/data/glove.6B.100d.txt''')
    

    config.emb_mat = emb_mat
    config.word_vocab_size = vocabulary_size
    config.char_vocab_size = char_vocabulary_size
    

    # with tf.name_scope("model"):
    #     model = Model(config, word2idx_dict, char2idx_dict)
    models = get_multi_models(config, word2idx_dict, char2idx_dict)

    # with tf.name_scope("trainer"):
    #     trainer = single_GPU_trainer(config, model)
    
    trainer = MultiGPUTrainer(config, models)

    
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    

    
    num_steps = config.num_steps
    global_step = 0


    train_data = DataSet(train_data_dict)
    train_data.init_with_ans_file('''/home/zhangs/RC/data/train_answers.json''', config.batch_size, 'train')



    train_writer = tf.summary.FileWriter('/home/zhangs/RC/data/nnlog_change_lr_without_a_batch', sess.graph)

    init = tf.global_variables_initializer()
    sess.run(init)

    batch_list = train_data.get_batch_list()
    batch_list_length = len(batch_list)
    batch_num = 5


    for i in range(config.num_epochs):

        # for i in range(int(math.ceil(batch_list_length/batch_num))):
        for i in range(int(math.ceil(batch_list_length/config.num_gpus))):
            # sub_batch_list = get_random_eles_from_list(batch_list, batch_num)
            sub_batch_list = get_random_eles_from_list(batch_list, config.num_gpus)

            global_step = sess.run(models[0].global_step) + 1
            print(global_step)
            if global_step == 10000:
                trainer.change_lr(0.2)
            loss, summary, train_op = trainer.step(sess, sub_batch_list, True)
            train_writer.add_summary(summary, global_step)
            print(loss)

            # for batch in sub_batch_list:
            #     global_step = sess.run(models[0].global_step) + 1  # +1 because all calculations are done after step
            #     get_summary = True
            #     print(global_step)

            #     loss, summary, train_op = trainer.step(sess, batch, get_summary=get_summary)
            #     train_writer.add_summary(summary, global_step)

            #     print(loss)
    

    '''start to evaluate via dev-set'''
    dev_data_dict = read_metadata('''/home/zhangs/RC/data/dev_v1.1.json''', 'dev')
    dev_data_dict_backup = read_metadata('''/home/zhangs/RC/data/dev_v1.1.json''', 'dev')
    dev_data   = DataSet(dev_data_dict)
    dev_data.init_without_ans(config.batch_size, 'dev')
    ans_list = dev_data.answers_list
    dev_batches = dev_data.get_batch_list()

    summaries = []
    for j, batch in enumerate(dev_batches):
        feed_dict = models[0].get_feed_dict(batch, None, False)
        yp, yp2 = sess.run([models[0].yp, models[0].yp2], feed_dict=feed_dict)   
        yp = get_y_index(yp)
        yp2= get_y_index(yp2)
        for i in range(len(batch['x'])):
            
            # print(dev_data_dict_backup['passages'][j*config.batch_size+i])
            
            wordss = batch['x'][i]
            # wo = wordss[yp[i][0]:yp2[i][0]+1]
            # wo[0] = wo[0][yp[i][1]:]
            # wo[len(wo)-1] = wo[len(wo)-1][:yp2[i][1]+1]
            # print(wo)
            try:
                summary = get_phrase(dev_data_dict_backup['passages'][j*config.batch_size+i], wordss, [yp[i], yp2[i]])
                summaries.append(summary)  
            except:
                print(yp[i])
                print(yp2[i])
                print(dev_data_dict_backup['passages'][j*config.batch_size+i])
                print(wordss)
    with open('''/home/zhangs/RC/data/ans_text.json''', 'w') as out:
        for i, summary in enumerate(summaries):
            di = {"answers": [summary], "query_id": dev_data_dict['query_ids'][i]}
            out.write(json.dumps(di) + '\n')
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
            

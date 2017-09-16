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

def main(config):
    # set_dirs(config)
    # with tf.device(config.device):
        if config.mode == 'train':
            _train(config)
        # elif config.mode == 'test':
        #     _test(config)
        # elif config.mode == 'forward':
        #     _forward(config)
        else:
            raise ValueError("invalid value for 'mode': {}".format(config.mode))


def _train(config):
    
    train_data_dict = read_metadata('''/home/zhangs/RC/data/train_v1.1.json''')
    # dev_data_dict = read_metadata('''/home/zhangs/RC/data/dev_v1.1.json''')
    char2idx_dict, char_vocabulary_size = get_char2idx(train_data_dict)
    

    word2idx_dict, emb_mat, vocabulary_size = get_word2idx_and_embmat('''/home/zhangs/RC/data/glove.6B.100d.txt''')
    

    ## Maximum passage size is about 2200
    config.max_num_sents = 22
    config.max_sent_size = 100
    config.max_ques_size = 20
    config.max_word_size = 20

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
    # dev_data   = DataSet(dev_data_dict)
    train_data.init_with_ans_file('''/home/zhangs/RC/data/train_answers.json''', config.batch_size, 'train')
    # dev_data.init_with_ans_file(path)



    train_writer = tf.summary.FileWriter('/home/zhangs/RC/data/nnlog_change_lr_without_a_batch', sess.graph)

    init = tf.global_variables_initializer()
    sess.run(init)

    batch_list = train_data.get_batch_list()
    batch_list_length = len(batch_list)
    batch_num = 5


    for i in range(config.num_epochs):

        # for i in range(int(math.ceil(batch_list_length/batch_num))):
        for i in range(int(math.ceil(batch_list_length/config.num_gpus)))
            # sub_batch_list = get_random_eles_from_list(batch_list, batch_num)
            sub_batch_list = get_random_eles_from_list(batch_list, config.num_gpus)

            global_step = sess.run(models[0].global_step) + 1
            print(global_step)
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
    



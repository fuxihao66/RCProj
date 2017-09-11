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
    with tf.device(config.device):
        if config.mode == 'train':
            _train(config)
        # elif config.mode == 'test':
        #     _test(config)
        # elif config.mode == 'forward':
        #     _forward(config)
        else:
            raise ValueError("invalid value for 'mode': {}".format(config.mode))


# def set_dirs(config):
#     # create directories
#     assert config.load or config.mode == 'train', "config.load must be True if not training"
#     if not config.load and os.path.exists(config.out_dir):
#         shutil.rmtree(config.out_dir)

#     config.save_dir = os.path.join(config.out_dir, "save")
#     config.log_dir = os.path.join(config.out_dir, "log")
#     config.eval_dir = os.path.join(config.out_dir, "eval")
#     config.answer_dir = os.path.join(config.out_dir, "answer")
#     if not os.path.exists(config.out_dir):
#         os.makedirs(config.out_dir)
#     if not os.path.exists(config.save_dir):
#         os.mkdir(config.save_dir)
#     if not os.path.exists(config.log_dir):
#         os.mkdir(config.log_dir)
#     if not os.path.exists(config.answer_dir):
#         os.mkdir(config.answer_dir)
#     if not os.path.exists(config.eval_dir):
#         os.mkdir(config.eval_dir)


# def _config_debug(config):
#     if config.debug:
#         config.num_steps = 2
#         config.eval_period = 1
#         config.log_period = 1
#         config.save_period = 1
#         config.val_num_batches = 2
#         config.test_num_batches = 2


def _train(config):
    
    train_data_dict = read_metadata('''/home/zhangs/RC/data/train_v1.1.json''')
    # dev_data_dict = read_metadata('''/home/zhangs/RC/data/dev_v1.1.json''')
    char2idx_dict, char_vocabulary_size = get_char2idx(train_data_dict)
    

    word2idx_dict, emb_mat, vocabulary_size = get_word2idx_and_embmat('''/home/zhangs/RC/data/glove.6B.100d.txt''')
    
    config.max_num_sents = 95
    config.max_sent_size = 100
    config.max_ques_size = 30
    config.max_word_size = 15

    config.emb_mat = emb_mat
    config.word_vocab_size = vocabulary_size
    config.char_vocab_size = char_vocabulary_size
    # construct model graph and variables (using default graph)
    # pprint(config.__flags, indent=2)
    with tf.name_scope("model"):
        model = Model(config, word2idx_dict, char2idx_dict)
    with tf.name_scope("trainer"):
        trainer = single_GPU_trainer(config, model)
    # evaluator = MultiGPUF1Evaluator(config, models, tensor_dict=model.tensor_dict if config.vis else None)
    # graph_handler = GraphHandler(config, model)  # controls all tensors and variables in the graph, including loading /saving

    # Variables
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    # graph_handler.initialize(sess)

    # Begin training
    # num_steps = config.num_steps or int(math.ceil(train_data.num_examples / (config.batch_size * config.num_gpus))) * config.num_epochs
    num_steps = config.num_steps
    global_step = 0


    train_data = DataSet(train_data_dict)
    # dev_data   = DataSet(dev_data_dict)
    train_data.init_with_ans_file('''/home/zhangs/RC/data/train_answers.json''', config.batch_size, 'train')
    # dev_data.init_with_ans_file(path)



    train_writer = tf.summary.FileWriter('/home/zhangs/RC/data/nnlog', sess.graph)
    # merged = tf.summary.merge_all()

    init = tf.global_variables_initializer()
    sess.run(init)

    batch_list = train_data.get_batch_list()
    batch_list_length = len(batch_list)
    batch_num = 10

    for i in range(config.num_epochs):
        
        for i in range(int(math.ceil(batch_list_length/batch_num))):
            sub_batch_list = get_random_eles_from_list(batch_list, batch_num)
            for batch in sub_batch_list:
                global_step = sess.run(model.global_step) + 1  # +1 because all calculations are done after step
                # get_summary = global_step % config.log_period == 0
                get_summary = True
                print(global_step)
                loss, summary, train_op = trainer.step(sess, batch, get_summary=get_summary)

                train_writer.add_summary(summary, global_step)
                print(loss)
    
    # for batch in tqdm(dev_data.get_batch_list()):
    #     sess.run(model.yp, model.yp2, feed_dict=model.get_feed_dict(batch, is_train=False))
        # print(yp, yp2)



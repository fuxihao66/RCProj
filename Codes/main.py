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
    
    config.max_num_sents = 20
    config.max_sent_size = 80
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

    train_data.init_with_ans_file('''/home/zhangs/RC/data/gen_ans.json''', config.batch_size)
    # dev_data.init_with_ans_file(path)



    train_writer = tf.summary.FileWriter('/home/zhangs/RC/data/nnlog', sess.graph)
    # merged = tf.summary.merge_all()

    init = tf.global_variables_initializer()
    sess.run(init)

    for batch in tqdm(train_data.get_batch_list()):
        global_step = sess.run(model.global_step) + 1  # +1 because all calculations are done after step
        # get_summary = global_step % config.log_period == 0
        get_summary = True
        loss, summary, train_op = trainer.step(sess, batch, get_summary=get_summary)

        train_writer.add_summary(summary, global_step)
        print(summary)
        print(loss)
        # if get_summary:
        #     graph_handler.add_summary(summary, global_step)

        # # occasional saving
        # if global_step % config.save_period == 0:
        #     graph_handler.save(sess, global_step=global_step)

        # if not config.eval:
        #     continue
        # # Occasional evaluation
        # if global_step % config.eval_period == 0:
        #     num_steps = math.ceil(dev_data.num_examples / (config.batch_size * config.num_gpus))
        #     if 0 < config.val_num_batches < num_steps:
        #         num_steps = config.val_num_batches
        #     e_train = evaluator.get_evaluation_from_batches(
        #         sess, tqdm(train_data.get_multi_batches(config.batch_size, config.num_gpus, num_steps=num_steps), total=num_steps)
        #     )
        #     graph_handler.add_summaries(e_train.summaries, global_step)
        #     e_dev = evaluator.get_evaluation_from_batches(
        #         sess, tqdm(dev_data.get_multi_batches(config.batch_size, config.num_gpus, num_steps=num_steps), total=num_steps))
        #     graph_handler.add_summaries(e_dev.summaries, global_step)

        #     if config.dump_eval:
        #         graph_handler.dump_eval(e_dev)
        #     if config.dump_answer:
        #         graph_handler.dump_answer(e_dev)
    # if global_step % config.save_period != 0:
    #     graph_handler.save(sess, global_step=global_step)


# def _test(config):
#     test_data = read_data(config, 'test', True)
#     update_config(config, [test_data])

#     _config_debug(config)

#     if config.use_glove_for_unk:
#         word2vec_dict = test_data.shared['lower_word2vec'] if config.lower_word else test_data.shared['word2vec']
#         new_word2idx_dict = test_data.shared['new_word2idx']
#         idx2vec_dict = {idx: word2vec_dict[word] for word, idx in new_word2idx_dict.items()}
#         new_emb_mat = np.array([idx2vec_dict[idx] for idx in range(len(idx2vec_dict))], dtype='float32')
#         config.new_emb_mat = new_emb_mat

#     pprint(config.__flags, indent=2)
#     models = get_multi_gpu_models(config)
#     model = models[0]
#     evaluator = MultiGPUF1Evaluator(config, models, tensor_dict=models[0].tensor_dict if config.vis else None)
#     graph_handler = GraphHandler(config, model)

#     sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
#     graph_handler.initialize(sess)
#     num_steps = math.ceil(test_data.num_examples / (config.batch_size * config.num_gpus))
#     if 0 < config.test_num_batches < num_steps:
#         num_steps = config.test_num_batches

#     e = None
#     for multi_batch in tqdm(test_data.get_multi_batches(config.batch_size, config.num_gpus, num_steps=num_steps, cluster=config.cluster), total=num_steps):
#         ei = evaluator.get_evaluation(sess, multi_batch)
#         e = ei if e is None else e + ei
#         if config.vis:
#             eval_subdir = os.path.join(config.eval_dir, "{}-{}".format(ei.data_type, str(ei.global_step).zfill(6)))
#             if not os.path.exists(eval_subdir):
#                 os.mkdir(eval_subdir)
#             path = os.path.join(eval_subdir, str(ei.idxs[0]).zfill(8))
#             graph_handler.dump_eval(ei, path=path)

#     print(e)
#     if config.dump_answer:
#         print("dumping answer ...")
#         graph_handler.dump_answer(e)
#     if config.dump_eval:
#         print("dumping eval ...")
#         graph_handler.dump_eval(e)


# def _forward(config):
#     assert config.load
#     test_data = read_data(config, config.forward_name, True)
#     update_config(config, [test_data])

#     _config_debug(config)

#     if config.use_glove_for_unk:
#         word2vec_dict = test_data.shared['lower_word2vec'] if config.lower_word else test_data.shared['word2vec']
#         new_word2idx_dict = test_data.shared['new_word2idx']
#         idx2vec_dict = {idx: word2vec_dict[word] for word, idx in new_word2idx_dict.items()}
#         new_emb_mat = np.array([idx2vec_dict[idx] for idx in range(len(idx2vec_dict))], dtype='float32')
#         config.new_emb_mat = new_emb_mat

#     pprint(config.__flags, indent=2)
#     models = get_multi_gpu_models(config)
#     model = models[0]
#     evaluator = ForwardEvaluator(config, model)
#     graph_handler = GraphHandler(config, model)  # controls all tensors and variables in the graph, including loading /saving

#     sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
#     graph_handler.initialize(sess)

#     num_batches = math.ceil(test_data.num_examples / config.batch_size)
#     if 0 < config.test_num_batches < num_batches:
#         num_batches = config.test_num_batches
#     e = evaluator.get_evaluation_from_batches(sess, tqdm(test_data.get_batches(config.batch_size, num_batches=num_batches), total=num_batches))
#     print(e)
#     if config.dump_answer:
#         print("dumping answer ...")
#         graph_handler.dump_answer(e, path=config.answer_path)
#     if config.dump_eval:
#         print("dumping eval ...")
#         graph_handler.dump_eval(e, path=config.eval_path)


def _get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path")
    return parser.parse_args()


class Config(object):
    def __init__(self, **entries):
        self.__dict__.update(entries)


def _run():
    args = _get_args()
    with open(args.config_path, 'r') as fh:
        config = Config(**json.load(fh))
        main(config)


if __name__ == "__main__":
    _run()
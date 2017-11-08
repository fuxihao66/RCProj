import tensorflow as tf
# from my.tensorflow import get_initializer
from utils.nn import softsel, get_logits, highway_network, multi_conv1d
from utils.rnn import bidirectional_dynamic_rnn, dynamic_rnn

# from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn 
# from utils.rnn_cell import SwitchableDropoutWrapper, AttentionCell
from tensorflow.python.ops.rnn_cell import DropoutWrapper
import numpy as np
from tensorflow.python.ops.rnn_cell import BasicLSTMCell, GRUCell
import itertools
from NewCell import *
from rnn_cell import *

def get_multi_models(config, word2idx_dict, char2idx_dict):
    models = []

    # the following line should be added for api>=12
    with tf.variable_scope(tf.get_variable_scope()):
        for gpu_idx in range(config.num_gpus):
            with tf.name_scope("model_{}".format(gpu_idx)) as scope, tf.device("/{}:{}".format(config.device_type, gpu_idx)):
                model = Model(config, scope, word2idx_dict, char2idx_dict)

                tf.get_variable_scope().reuse_variables()
                models.append(model)

    return models
class Model:
    def __init__(self, config, scope, word2idx_dict, char2idx_dict):

        self.scope = scope

        self.config = config

        self.global_step = tf.get_variable('global_step', shape=[], dtype='int32',
                                           initializer=tf.constant_initializer(0), trainable=False)
        
        self.word2idx_dict = word2idx_dict
        self.char2idx_dict = char2idx_dict
        # x means the indexes of words of para in the emb_dict
        '''
        x [index_in_batch, index_in_sentence, index_of_word_in_this_sent]
        cx [index_in_batch, .., .., index_of_char_in_word]
        q [index_in_batch, index_of_word_in_q]

        y [index_in_batch, index_in_sent, index_of_word_in_this_sent]
        '''
        self.x = tf.placeholder('int32', [config.batch_size, None, None])
        self.cx = tf.placeholder('int32', [config.batch_size, None, None, config.max_word_size])
        self.q = tf.placeholder('int32', [config.batch_size, None])
        self.cq = tf.placeholder('int32', [config.batch_size, None, config.max_word_size])
        self.y = tf.placeholder('bool', [config.batch_size, None, None])
        self.y2 = tf.placeholder('bool', [config.batch_size, None, None])


        '''an optimization: 让非mask的在softmax后非常小'''
        self.x_mask = tf.placeholder('bool', [config.batch_size, None, None], name='x_mask')
        self.q_mask = tf.placeholder('bool', [config.batch_size, None], name='q_mask')
        self.is_train = tf.placeholder('bool', [], name='is_train')

        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        self.emb_mat = tf.placeholder('float', [None, 100])
        
        self.tensor_dict = {}
        self.loss = None
        self.var_list = None
        self.build_forward()

        # self.build_loss()
        # self.var_ema = None
        # self.build_var_ema()
        
        # if config.mode == 'train':
        #     self.build_ema()

        self.summary = tf.summary.merge_all()
        print(1)
    def get_lr(self):
        return self.learning_rate
    def build_forward(self):
        config = self.config
        N, M, JX, JQ, VW , VC, d, W = \
            config.batch_size, config.max_num_sents, config.max_sent_size, \
            config.max_ques_size, config.word_vocab_size, config.char_vocab_size, config.hidden_size, \
            config.max_word_size
        JX = tf.shape(self.x)[2]
        JQ = tf.shape(self.q)[1]
        M = tf.shape(self.x)[1]
        dc, dw, dco = config.char_emb_size, config.word_emb_size, config.char_out_size

        with tf.variable_scope("embedding"):
       
            with tf.variable_scope("emb_var"), tf.device("/cpu:0"):
                char_emb_mat = tf.get_variable("char_emb_mat", shape=[VC, dc], dtype='float')

            with tf.variable_scope("char"), tf.device("/cpu:0"):
                Acx = tf.nn.embedding_lookup(char_emb_mat, self.cx)  # [N, M, JX, W, dc]
                Acq = tf.nn.embedding_lookup(char_emb_mat, self.cq)  # [N, JQ, W, dc]
                Acx = tf.reshape(Acx, [-1, JX, W, dc])
                Acq = tf.reshape(Acq, [-1, JQ, W, dc])

                filter_sizes = list(map(int, config.out_channel_dims.split(',')))
                heights = list(map(int, config.filter_heights.split(',')))
                assert sum(filter_sizes) == dco, (filter_sizes, dco)
                with tf.variable_scope("conv"):
                    xx = multi_conv1d(Acx, filter_sizes, heights, "VALID",  self.is_train, config.keep_prob, scope="xx")
                    if config.share_cnn_weights:
                        tf.get_variable_scope().reuse_variables()
                        qq = multi_conv1d(Acq, filter_sizes, heights, "VALID", self.is_train, config.keep_prob, scope="xx")
                    else:
                        qq = multi_conv1d(Acq, filter_sizes, heights, "VALID", self.is_train, config.keep_prob, scope="qq")
                    xx = tf.reshape(xx, [-1, M, JX, dco])
                    qq = tf.reshape(qq, [-1, JQ, dco])

        
            with tf.name_scope("word"):
                Ax = tf.nn.embedding_lookup(self.emb_mat, self.x)  # [N, M, JX, d]
                Aq = tf.nn.embedding_lookup(self.emb_mat, self.q)  # [N, JQ, d]
                self.tensor_dict['x'] = Ax
                self.tensor_dict['q'] = Aq
            if config.use_char_emb:
                xx = tf.concat([xx, Ax], 3)  # [N, M, JX, di]
                qq = tf.concat([qq, Aq], 2)  # [N, JQ, di]
            else:
                xx = Ax
                qq = Aq

        self.tensor_dict['xx'] = xx
        self.tensor_dict['qq'] = qq



        

        '''x_len means the length of sequences '''
        x_len = tf.reduce_sum(tf.cast(self.x_mask, 'int32'), 2)  # [N, M]
        q_len = tf.reduce_sum(tf.cast(self.q_mask, 'int32'), 1)  # [N]

        with tf.variable_scope("synthesis"):
            # cell = GRUCell(d)
            cell = GeneGRUCell(d, qq, 200)
            output, state = dynamic_rnn(cell, qq, q_len, dtype='float', scope='hq')
            
            d, cont = tf.split(output, [100, 200], 2)
            
            
            # print(fw_d)
            # print(cont)
            # print(fw_hq)
            # h_q = tf.concat([fw_hq, bw_hq], 2)
            # (fw_hp, bw_hp), (_,_) = bidirectional_dynamic_rnn(cell, cell, xx, x_len, dtype='float', scope='hp')
            # h_p = tf.concat([fw_hp, bw_hp], 3)

            # gene_cell = GeneCell(cell, h_q)
            # init_d0 = tf.nn.tanh(_linear([],d, True))
            # d_t,  = dynamic_rnn(gene_cell, xx, initial_state=init_d0 ,dtype='float', scope='d')
            d_t = tf.reshape(d_t, [-1, 100])
            xx = tf.reshape(xx, [-1, 200])
            cont = tf.reshape(cont, [-1, 200])

            r_t = _linear([xx, cont, d_t], 2*d, False)
            r_t = tf.reshape(r_t, [config.batch_size, -1, 2*d])

            m_t = get_maxout(r_t)   #[batch_size, seq_len, 100]
            m_t = tf.reshape(m_t, shape=[-1, 100])
            W_0 = tf.get_variable(name='W_o', shape=[d,voca_size], dtype='float')
            flat_p = tf.matmul(m_t, W_0)
            p = tf.reshape(flat_p, shape=[batch_size, -1, voca_size]) #[batch_size, seq_len, voca_size]
            p = tf.nn.softmax(p, 2)
        


    def build_loss(self):
        
        config = self.config
        JX = tf.shape(self.x)[2]
        M = tf.shape(self.x)[1]
        JQ = tf.shape(self.q)[1]
        loss_mask = tf.reduce_max(tf.cast(self.q_mask, 'float'), 1)

        '''compute the cross entropy loss with y and logit'''
        losses = tf.nn.softmax_cross_entropy_with_logits(
            logits=self.logits, labels=tf.cast(tf.reshape(self.y, [-1, M * JX]), 'float'))
        ce_loss = tf.reduce_mean(loss_mask * losses)
        tf.add_to_collection('losses', ce_loss)
        ce_loss2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.logits2, labels=tf.cast(tf.reshape(self.y2, [-1, M * JX]), 'float')))
        tf.add_to_collection("losses", ce_loss2)

        self.loss = tf.add_n(tf.get_collection('losses', scope=self.scope), name='loss')
        # self.loss = tf.add_n(tf.get_collection('losses'), name='loss')
        tf.summary.scalar(self.loss.op.name, self.loss)
        tf.add_to_collection('ema/scalar', self.loss)

        




    def build_ema(self):
        
        self.ema = tf.train.ExponentialMovingAverage(self.config.decay)
        ema = self.ema
        
        tensors = tf.get_collection("ema/scalar", scope=self.scope) + tf.get_collection("ema/vector", scope=self.scope)
        with tf.name_scope(None):
            ema_op = ema.apply(tensors)
        for var in tf.get_collection("ema/scalar", scope=self.scope):
            ema_var = ema.average(var)
            tf.summary.scalar(ema_var.op.name, ema_var)
        for var in tf.get_collection("ema/vector", scope=self.scope):
            ema_var = ema.average(var)
            tf.summary.histogram(ema_var.op.name, ema_var)

        with tf.control_dependencies([ema_op]):
            self.loss = tf.identity(self.loss)
    
    def get_loss(self):
        return self.loss

    def get_global_step(self):
        return self.global_step

    def get_var_list(self):
        return self.var_list
    def build_var_ema(self):
        self.var_ema = tf.train.ExponentialMovingAverage(self.config.var_decay)
        ema = self.var_ema
        ema_op = ema.apply(tf.trainable_variables())
        with tf.control_dependencies([ema_op]):
            self.loss = tf.identity(self.loss)

    
    '''
    given a batch, generate a feed dict to training
    Format:
    {
        x: [[ [1,2,..],[3,4,..],[5,6,..] ], [], [], []]  
    }
    '''
    def get_feed_dict(self, batch, lr, is_train, supervised=True):

        config = self.config
        N, M, JX, JQ, VW, VC, d, W = \
            config.batch_size, config.max_num_sents, config.max_sent_size, \
            config.max_ques_size, config.word_vocab_size, config.char_vocab_size, config.hidden_size, config.max_word_size
        feed_dict = {}

        x = np.zeros([N, M, JX], dtype='int32')
        cx = np.zeros([N, M, JX, W], dtype='int32')
        x_mask = np.zeros([N, M, JX], dtype='bool')
        q = np.zeros([N, JQ], dtype='int32')
        cq = np.zeros([N, JQ, W], dtype='int32')
        q_mask = np.zeros([N, JQ], dtype='bool')

        feed_dict[self.x] = x
        feed_dict[self.x_mask] = x_mask
        feed_dict[self.cx] = cx
        feed_dict[self.q] = q
        feed_dict[self.cq] = cq
        feed_dict[self.q_mask] = q_mask
        feed_dict[self.is_train] = is_train
        feed_dict[self.learning_rate] = lr
        X = batch['x']
        CX = batch['cx']

        if supervised:
            y = np.zeros([N, M, JX], dtype='bool')
            y2 = np.zeros([N, M, JX], dtype='bool')
            feed_dict[self.y] = y
            feed_dict[self.y2] = y2

            for i, yi in enumerate(batch['y']):  
                if yi[0][1] < JX and yi[1][1] < JX and yi[0][0] < M and yi[1][0] < M:  
                    [j, k] = yi[0]
                    [j2, k2] = yi[1]
                    
                else:
                    [j, k] = [0, 0]
                    [j2, k2] = [0,0]
                    
                y[i, j, k] = True
                y2[i, j2, k2] = True

        def _get_word(word):
            d = self.word2idx_dict
            for each in (word, word.lower(), word.capitalize(), word.upper()):
                if each in d:
                    return d[each]
            return 0

        def _get_char(char):
            d = self.char2idx_dict
            if char in d:
                return d[char]
            return 0

        for i, xi in enumerate(X):
            if self.config.squash:
                xi = [list(itertools.chain(*xi))]
            for j, xij in enumerate(xi):
                if j == config.max_num_sents:
                    break
                for k, xijk in enumerate(xij):
                    if k == config.max_sent_size:
                        break
                    each = _get_word(xijk)
                    assert isinstance(each, int), each
                    x[i, j, k] = each
                    x_mask[i, j, k] = True

        for i, cxi in enumerate(CX):
            if self.config.squash:
                cxi = [list(itertools.chain(*cxi))]
            for j, cxij in enumerate(cxi):
                if j == config.max_num_sents:
                    break
                for k, cxijk in enumerate(cxij):
                    if k == config.max_sent_size:
                        break
                    for l, cxijkl in enumerate(cxijk):
                        if l == config.max_word_size:
                            break
                        cx[i, j, k, l] = _get_char(cxijkl)

        for i, qi in enumerate(batch['q']):
            for j, qij in enumerate(qi):
                if j == config.max_ques_size:
                    break
                q[i, j] = _get_word(qij)
                q_mask[i, j] = True

        for i, cqi in enumerate(batch['cq']):
            for j, cqij in enumerate(cqi):
                if j == config.max_ques_size:
                    break
                for k, cqijk in enumerate(cqij):
                    cq[i, j, k] = _get_char(cqijk)
                    if k + 1 == config.max_word_size:
                        break

        return feed_dict

    


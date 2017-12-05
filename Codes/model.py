import tensorflow as tf
# from my.tensorflow import get_initializer
from utils.nn import softsel, get_logits, highway_network, multi_conv1d
from utils.rnn import bidirectional_dynamic_rnn
# from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn 
from utils.rnn_cell import SwitchableDropoutWrapper, AttentionCell
from tensorflow.python.ops.rnn_cell import DropoutWrapper
import numpy as np
from tensorflow.python.ops.rnn_cell import BasicLSTMCell
import itertools


def get_multi_models(config, word2idx_dict, char2idx_dict):
    models = []

    # the following line should be added for api>=12
    with tf.variable_scope(tf.get_variable_scope()):
        for gpu_idx in range(config.num_gpus):
            with tf.name_scope("model_{}".format(gpu_idx)) as scope, tf.device("/{}:{}".format(config.device_type, gpu_idx)):
                model = Model(config, word2idx_dict, char2idx_dict, scope=scope)

                tf.get_variable_scope().reuse_variables()
                models.append(model)

    return models
class Model:
    def __init__(self, config, word2idx_dict, char2idx_dict, scope=None):

        self.scope = scope

        self.config = config
        self.emb_mat = config.emb_mat
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
        self.x = tf.placeholder('int32', [config.batch_size, None])
        self.cx = tf.placeholder('int32', [config.batch_size, None, config.max_word_size])
        self.q = tf.placeholder('int32', [config.batch_size, None])
        self.cq = tf.placeholder('int32', [config.batch_size, None, config.max_word_size])
        self.y = tf.placeholder('bool', [config.batch_size, None])
        self.y2 = tf.placeholder('bool', [config.batch_size, None])


        '''an optimization: 让非mask的在softmax后非常小'''
        self.x_mask = tf.placeholder('bool', [config.batch_size, None], name='x_mask')
        self.q_mask = tf.placeholder('bool', [config.batch_size, None], name='q_mask')
        self.is_train = tf.placeholder('bool', [], name='is_train')

        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        # self.emb_mat = tf.placeholder('float', [None, word_emb_size])
        
        self.tensor_dict = {}
        self.loss = None
        self.var_list = None
        self.build_forward()

        self.build_loss()
        self.var_ema = None
        self.build_var_ema()
        
        if config.mode == 'train':
            self.build_ema()

        self.summary = tf.summary.merge_all()
        print(1)
    def get_lr(self):
        return self.learning_rate
    def build_forward(self):
        config = self.config
        N, JX, JQ, VW , VC, d, W = \
            config.batch_size,  config.max_sent_size, \
            config.max_ques_size, config.word_vocab_size, config.char_vocab_size, config.hidden_size, \
            config.max_word_size
        JX = tf.shape(self.x)[1]
        JQ = tf.shape(self.q)[1]
        
        dc, dw, dco = config.char_emb_size, config.word_emb_size, config.char_out_size

        with tf.variable_scope("embedding"):
       
            with tf.variable_scope("emb_var"), tf.device("/cpu:0"):
                char_emb_mat = tf.get_variable("char_emb_mat", shape=[VC, dc], dtype='float')

            with tf.variable_scope("char"), tf.device("/cpu:0"):
                Acx = tf.nn.embedding_lookup(char_emb_mat, self.cx)  # [N,  JX, W, dc]
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
                    xx = tf.reshape(xx, [-1, JX, dco])
                    qq = tf.reshape(qq, [-1, JQ, dco])

        
            with tf.name_scope("word"):
                Ax = tf.nn.embedding_lookup(self.emb_mat, self.x)  # [N, M, JX, d]
                Aq = tf.nn.embedding_lookup(self.emb_mat, self.q)  # [N, JQ, d]
                self.tensor_dict['x'] = Ax
                self.tensor_dict['q'] = Aq
            if config.use_char_emb:
                xx = tf.concat([xx, Ax], 2)  # [N, JX, di]
                qq = tf.concat([qq, Aq], 2)  # [N, JQ, di]
            else:
                xx = Ax
                qq = Aq

        self.tensor_dict['xx'] = xx
        self.tensor_dict['qq'] = qq


        '''x_len means the length of sequences '''
        x_len = tf.reduce_sum(tf.cast(self.x_mask, 'int32'), 1)  # [N, M]
        q_len = tf.reduce_sum(tf.cast(self.q_mask, 'int32'), 1)  # [N]

        with tf.variable_scope("Encoding"):
            cell = BasicLSTMCell(d, state_is_tuple=True)
            encoding_cell = SwitchableDropoutWrapper(cell, self.is_train, input_keep_prob=config.input_keep_prob)

            (fw_u, bw_u), ((_, fw_u_f), (_, bw_u_f)) = bidirectional_dynamic_rnn(encoding_cell, encoding_cell, qq, q_len, dtype='float', scope='u1')  # [N, J, d], [N, d]
            u = tf.concat([fw_u, bw_u], 2)
            if config.share_lstm_weights:
                tf.get_variable_scope().reuse_variables()
                (fw_h, bw_h), _ = bidirectional_dynamic_rnn(cell, cell, xx, x_len, dtype='float', scope='u1')  # [N, JX, 2d]
                h = tf.concat([fw_h, bw_h], 2)  # [N, JX, 2d]
            else:
                (fw_h, bw_h), _ = bidirectional_dynamic_rnn(cell, cell, xx, x_len, dtype='float', scope='h1')  # [N,  JX, 2d]
                h = tf.concat([fw_h, bw_h], 2)  # [N, h = tf.concat([fw_h, bw_h], 2)  # [N, JX, 2d]
            self.tensor_dict['u'] = u
            self.tensor_dict['h'] = h

        with tf.variable_scope("main"):
            
            if config.dynamic_att:
                p0 = h
                u = tf.reshape(tf.tile(tf.expand_dims(u, 1), [1, M, 1, 1]), [N * M, JQ, 2 * d])
                q_mask = tf.reshape(tf.tile(tf.expand_dims(self.q_mask, 1), [1, M, 1]), [N * M, JQ])
                first_cell = AttentionCell(cell, u, mask=q_mask, mapper='sim',
                                           input_keep_prob=self.config.input_keep_prob, is_train=self.is_train)
            else:
                ## G
                p0 = attention_layer(config, self.is_train, h, u, h_mask=self.x_mask, u_mask=self.q_mask, scope="p0", tensor_dict=self.tensor_dict)
                cell = BasicLSTMCell(d, state_is_tuple=True)
                first_cell = SwitchableDropoutWrapper(cell, self.is_train, input_keep_prob=config.input_keep_prob)
    

            '''the following can be simplified, by using multi-layer rnn'''
            ## 2 layers of bi rnn
            (fw_g0, bw_g0), _ = bidirectional_dynamic_rnn(first_cell, first_cell, p0, x_len, dtype='float', scope='g0')  # [N, JX, 2d]
            g0 = tf.concat([fw_g0, bw_g0], 2)

            cell = BasicLSTMCell(d, state_is_tuple=True)
            first_cell = SwitchableDropoutWrapper(cell, self.is_train, input_keep_prob=config.input_keep_prob)

            (fw_g1, bw_g1), _ = bidirectional_dynamic_rnn(first_cell, first_cell, g0, x_len, dtype='float', scope='g1')  # [N, JX, 2d]
            ##M
            g1 = tf.concat([fw_g1, bw_g1], 2)

            logits = get_logits([g1, p0], d, True, wd=config.wd, input_keep_prob=config.input_keep_prob,
                                mask=self.x_mask, is_train=self.is_train, func=config.answer_func, scope='logits1')





            a1i = softsel(tf.reshape(g1, [N, JX, 2 * d]), tf.reshape(logits, [N, JX]))
            a1i = tf.tile(tf.expand_dims(a1i, 1), [1, JX, 1])

            cell = BasicLSTMCell(d, state_is_tuple=True)
            M2_operate_cell = SwitchableDropoutWrapper(cell, self.is_train, input_keep_prob=config.input_keep_prob)

            (fw_g2, bw_g2), _ = bidirectional_dynamic_rnn(M2_operate_cell, M2_operate_cell, tf.concat([p0, g1, a1i, g1 * a1i], 2),
                                                          x_len, dtype='float', scope='g2')  # [N, M, JX, 2d]
            ## M^2
            g2 = tf.concat([fw_g2, bw_g2], 2)

            logits2 = get_logits([g2, p0], d, True, wd=config.wd, input_keep_prob=config.input_keep_prob,
                                 mask=self.x_mask,
                                 is_train=self.is_train, func=config.answer_func, scope='logits2')

            flat_logits = tf.reshape(logits, [-1, JX])
            yp = tf.nn.softmax(flat_logits)  # [-1, JX]
            flat_logits2 = tf.reshape(logits2, [-1, JX])
            yp2 = tf.nn.softmax(flat_logits2)


            self.tensor_dict['g1'] = g1
            self.tensor_dict['g2'] = g2

            self.logits = flat_logits
            self.logits2 = flat_logits2
            self.yp = yp
            self.yp2 = yp2

    def get_loss(self):
        return self.loss

    def get_global_step(self):
        return self.global_step

    def get_var_list(self):
        return self.var_list

    def build_loss(self):
        
        config = self.config
        JX = tf.shape(self.x)[1]
        JQ = tf.shape(self.q)[1]
        loss_mask = tf.reduce_max(tf.cast(self.q_mask, 'float'), 1)

        '''compute the cross entropy loss with y and logit'''
        losses = tf.nn.softmax_cross_entropy_with_logits(
            logits=self.logits, labels=tf.cast(tf.reshape(self.y, [-1, JX]), 'float'))
        ce_loss = tf.reduce_mean(loss_mask * losses)
        tf.add_to_collection('losses', ce_loss)
        ce_loss2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.logits2, labels=tf.cast(tf.reshape(self.y2, [-1, JX]), 'float')))
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
        N, JX, JQ, VW, VC, d, W = \
            config.batch_size, config.max_sent_size, \
            config.max_ques_size, config.word_vocab_size, config.char_vocab_size, config.hidden_size, config.max_word_size
        feed_dict = {}

        x = np.zeros([N, JX], dtype='int32')
        cx = np.zeros([N, JX, W], dtype='int32')
        x_mask = np.zeros([N, JX], dtype='bool')
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
            y = np.zeros([N, JX], dtype='bool')
            y2 = np.zeros([N, JX], dtype='bool')
            feed_dict[self.y] = y
            feed_dict[self.y2] = y2

            for i, yi in enumerate(batch['y']):  
                j = yi[0]
                j2 = yi[1]
                
                if j< config.max_sent_size and j2 < config.max_sent_size:
                    y[i, j] = True
                    y2[i, j2] = True
                else:
                    y[i, 0] = True
                    y2[i, 0] = True

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
            
            for k, xik in enumerate(xi):
                if k == config.max_sent_size:
                    break
                each = _get_word(xik)
                assert isinstance(each, int), each
                x[i, k] = each
                x_mask[i, k] = True

        for i, cxi in enumerate(CX):
                for k, cxik in enumerate(cxi):
                    if k == config.max_sent_size:
                        break
                    for l, cxikl in enumerate(cxik):
                        if l == config.max_word_size:
                            break
                        cx[i, k, l] = _get_char(cxikl)

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


'''
the functions below are implemented with the method mentioned in the paper
'''
def bi_attention(config, is_train, h, u, h_mask=None, u_mask=None, scope=None, tensor_dict=None):
    with tf.variable_scope(scope or "bi_attention"):
        JX = tf.shape(h)[1]
        JQ = tf.shape(u)[1]
        h_aug = tf.tile(tf.expand_dims(h, 2), [1, 1, JQ, 1])
        u_aug = tf.tile(tf.expand_dims(u, 1), [1, JX, 1, 1])#do expand_dims 1 time less
        if h_mask is None:
            hu_mask = None
        else:
            h_mask_aug = tf.tile(tf.expand_dims(h_mask, 2), [1, 1, JQ])
            u_mask_aug = tf.tile(tf.expand_dims(u_mask, 1), [1, JX, 1])#do expand_dims 1 time less
            hu_mask = h_mask_aug & u_mask_aug

        #S
        u_logits = get_logits([h_aug, u_aug], None, True, wd=config.wd, mask=hu_mask,
                              is_train=is_train, func=config.logit_func, scope='u_logits')  # [N, M, JX, JQ]
        
        #对logits做softmax，aug做加权和
        u_a = softsel(u_aug, u_logits)  # [N, JX, d]
        h_a = softsel(h, tf.reduce_max(u_logits, 2))  # [N, d]
        print(h_a)
        h_a = tf.tile(tf.expand_dims(h_a, 1), [1, JX, 1])


        if tensor_dict is not None:
            a_u = tf.nn.softmax(u_logits)  # [N, JX, JQ]
            a_h = tf.nn.softmax(tf.reduce_max(u_logits, 2))
            tensor_dict['a_u'] = a_u
            tensor_dict['a_h'] = a_h
            variables = tf.get_collection(tf.GraphKeys.VARIABLES, scope=tf.get_variable_scope().name)
            for var in variables:
                tensor_dict[var.name] = var

        return u_a, h_a


def attention_layer(config, is_train, h, u, h_mask=None, u_mask=None, scope=None, tensor_dict=None):
    with tf.variable_scope(scope or "attention_layer"):
        JX = tf.shape(h)[1]
        # M = tf.shape(h)[1]
        JQ = tf.shape(u)[1]
        if config.q2c_att or config.c2q_att:
            u_a, h_a = bi_attention(config, is_train, h, u, h_mask=h_mask, u_mask=u_mask, tensor_dict=tensor_dict)
        # if not config.c2q_att:
        #     u_a = tf.tile(tf.expand_dims(tf.expand_dims(tf.reduce_mean(u, 1), 1), 1), [1, JX, 1])
        if config.q2c_att:
            p0 = tf.concat([h, u_a, h * u_a, h * h_a], 2)  
        else:
            p0 = tf.concat([h, u_a, h * u_a], 2)
        return p0



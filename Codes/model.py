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
# import tensorflow.python.ops.rnn
# def get_cell_type(model):
#     if model == 'rnn':
#         cell_type = rnn.BasicRNNCell
#     elif model == 'lstm':
#         cell_type = rnn.BasicLSTMCell
#     elif model == 'gru':
#         cell_type = rnn.GRUCell
#     else:
#         raise Exception('model type not supported:{}'.format(model))
#     return cell_type

# def muliti_layers_init(rnn_size, layer_nums, cell_type):
#     #every element stands for a layer
#     cells = []
#     for _ in range(layer_nums):
#         cell = cell_type(rnn_size)
#         cells.append(cell)
#     return cells
# def single_layer_init(rnn_size, cell_type):
#     cell = cell_type(rnn_size)
#     return cell 

# def Rnn(extension=None, cell_fw, cell_bw=None, inputs, sequence_lengths=None, init_fw=None, init_bw=None, dtype=None):
#     if extension == 'bi':
#         return nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs, sequence_lengths, initial_state_fw=init_fw, initial_state_bw=init_bw dtype=dtype)
#     elif extension == None:
#         return nn.dynamic_rnn(cell_fw, inputs, sequence_lengths, initial_state=init_fw, dtype=dtype)
#     else :
#         raise Exception('Extension type error:{}'.format(extension))


'''
parameters{
    batch_size,
    learning_rate,
    dropout_rate,
    dtype,
    lr
}
'''

# def get_multi_gpu_models(config):
#     models = []
#     for gpu_idx in range(config.num_gpus):
#         with tf.name_scope("model_{}".format(gpu_idx)) as scope, tf.device("/{}:{}".format(config.device_type, gpu_idx)):
#             model = Model(config, scope, rep=gpu_idx == 0)
#             tf.get_variable_scope().reuse_variables()
#             models.append(model)
#     return models
class Model:
    def __init__(self, config, word2idx_dict, char2idx_dict):

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
        self.x = tf.placeholder('int32', [config.batch_size, None, None])
        self.cx = tf.placeholder('int32', [config.batch_size, None, None, config.max_word_size])
        self.q = tf.placeholder('int32', [config.batch_size, None])
        self.cq = tf.placeholder('int32', [config.batch_size, None, config.max_word_size])
        self.y = tf.placeholder('bool', [config.batch_size, None, None])
        self.y2 = tf.placeholder('bool', [config.batch_size, None, None])

        self.x_mask = tf.placeholder('bool', [config.batch_size, None, None], name='x_mask')
        self.q_mask = tf.placeholder('bool', [config.batch_size, None], name='q_mask')
        self.is_train = tf.placeholder('bool', [], name='is_train')
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

            with tf.variable_scope("char"):
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

           
            # with tf.variable_scope("emb_var"), tf.device("/cpu:0"):
            #     if config.mode == 'train':
            #         word_emb_mat = tf.get_variable("word_emb_mat", dtype='float', shape=[VW, dw], initializer=get_initializer(config.emb_mat))
            #     else:
            #         word_emb_mat = tf.get_variable("word_emb_mat", shape=[VW, dw], dtype='float')
            #     if config.use_glove_for_unk:
            #         word_emb_mat = tf.concat(0, [word_emb_mat, self.new_emb_mat])
            print('start word embedding')
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



        cell = BasicLSTMCell(d, state_is_tuple=True)
        d_cell = SwitchableDropoutWrapper(cell, self.is_train, input_keep_prob=config.input_keep_prob)
        # d_cell = DropoutWrapper(cell, input_keep_prob=config.input_keep_prob)

        '''x_len means the length of sequences '''
        x_len = tf.reduce_sum(tf.cast(self.x_mask, 'int32'), 2)  # [N, M]
        q_len = tf.reduce_sum(tf.cast(self.q_mask, 'int32'), 1)  # [N]

        with tf.variable_scope("Encoding"):
            (fw_u, bw_u), ((_, fw_u_f), (_, bw_u_f)) = bidirectional_dynamic_rnn(d_cell, d_cell, qq, q_len, dtype='float', scope='u1')  # [N, J, d], [N, d]
            u = tf.concat([fw_u, bw_u], 2)
            if config.share_lstm_weights:
                tf.get_variable_scope().reuse_variables()
                (fw_h, bw_h), _ = bidirectional_dynamic_rnn(cell, cell, xx, x_len, dtype='float', scope='u1')  # [N, M, JX, 2d]
                h = tf.concat([fw_h, bw_h], 3)  # [N, M, JX, 2d]
            else:
                (fw_h, bw_h), _ = bidirectional_dynamic_rnn(cell, cell, xx, x_len, dtype='float', scope='h1')  # [N, M, JX, 2d]
                h = tf.concat([fw_h, bw_h], 3)  # [N, M, JX, 2d]
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
                p0 = attention_layer(config, self.is_train, h, u, h_mask=self.x_mask, u_mask=self.q_mask, scope="p0", tensor_dict=self.tensor_dict)


                tcell = BasicLSTMCell(d, state_is_tuple=True)
                first_cell = SwitchableDropoutWrapper(tcell, self.is_train, input_keep_prob=config.input_keep_prob)
                # first_cell = d_cell

            
            (fw_g0, bw_g0), _ = bidirectional_dynamic_rnn(first_cell, first_cell, p0, x_len, dtype='float', scope='g0')  # [N, M, JX, 2d]
            g0 = tf.concat([fw_g0, bw_g0], 3)

            ttcell = BasicLSTMCell(d, state_is_tuple=True)
            first_cell = SwitchableDropoutWrapper(ttcell, self.is_train, input_keep_prob=config.input_keep_prob)

            (fw_g1, bw_g1), _ = bidirectional_dynamic_rnn(first_cell, first_cell, g0, x_len, dtype='float', scope='g1')  # [N, M, JX, 2d]
            g1 = tf.concat([fw_g1, bw_g1], 3)

            logits = get_logits([g1, p0], d, True, wd=config.wd, input_keep_prob=config.input_keep_prob,
                                mask=self.x_mask, is_train=self.is_train, func=config.answer_func, scope='logits1')
            a1i = softsel(tf.reshape(g1, [N, M * JX, 2 * d]), tf.reshape(logits, [N, M * JX]))
            a1i = tf.tile(tf.expand_dims(tf.expand_dims(a1i, 1), 1), [1, M, JX, 1])


            ttcell = BasicLSTMCell(d, state_is_tuple=True)
            d_cell = SwitchableDropoutWrapper(ttcell, self.is_train, input_keep_prob=config.input_keep_prob)

            (fw_g2, bw_g2), _ = bidirectional_dynamic_rnn(d_cell, d_cell, tf.concat([p0, g1, a1i, g1 * a1i], 3),
                                                          x_len, dtype='float', scope='g2')  # [N, M, JX, 2d]
            g2 = tf.concat([fw_g2, bw_g2], 3)
            logits2 = get_logits([g2, p0], d, True, wd=config.wd, input_keep_prob=config.input_keep_prob,
                                 mask=self.x_mask,
                                 is_train=self.is_train, func=config.answer_func, scope='logits2')

            flat_logits = tf.reshape(logits, [-1, M * JX])
            flat_yp = tf.nn.softmax(flat_logits)  # [-1, M*JX]
            yp = tf.reshape(flat_yp, [-1, M, JX])
            flat_logits2 = tf.reshape(logits2, [-1, M * JX])
            flat_yp2 = tf.nn.softmax(flat_logits2)
            yp2 = tf.reshape(flat_yp2, [-1, M, JX])

            self.tensor_dict['g1'] = g1
            self.tensor_dict['g2'] = g2

            self.logits = flat_logits
            self.logits2 = flat_logits2
            self.yp = yp
            self.yp2 = yp2

            
        # with tf.variable_scope('encoding'):
        #     cell_type = get_cell_type(config.encoding_cell_type)
        #     cell = cell_type(config.hidden_size_encoding, state_is_tuple=True)
        # with tf.variable_scope('interaction'):

        # with tf.variable_scope('answer'):
    def get_loss(self):
        return self.loss

    def get_global_step(self):
        return self.global_step

    def get_var_list(self):
        return self.var_list

    def build_loss(self):
        with tf.name_scope('loss'):
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

            # self.loss = tf.add_n(tf.get_collection('losses', scope=self.scope), name='loss')
            self.loss = tf.add_n(tf.get_collection('losses'), name='loss')
            tf.summary.scalar(self.loss.op.name, self.loss)
            tf.add_to_collection('ema/scalar', self.loss)
    def build_ema(self):
        self.ema = tf.train.ExponentialMovingAverage(self.config.decay)
        ema = self.ema
        tensors = tf.get_collection("ema/scalar") + tf.get_collection("ema/vector")
        ema_op = ema.apply(tensors)
        for var in tf.get_collection("ema/scalar"):
            ema_var = ema.average(var)
            tf.summary.scalar(ema_var.op.name, ema_var)
        for var in tf.get_collection("ema/vector"):
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
    def get_feed_dict(self, batch, is_train):

        config = self.config
        N, M, JX, JQ, VW, VC, d, W = \
            config.batch_size, config.max_num_sents, config.max_sent_size, \
            config.max_ques_size, config.word_vocab_size, config.char_vocab_size, config.hidden_size, config.max_word_size
        feed_dict = {}

        # if config.len_opt:
        #     """
        #     Note that this optimization results in variable GPU RAM usage (i.e. can cause OOM in the middle of training.)
        #     First test without len_opt and make sure no OOM, and use len_opt
        #     """
        #     if sum(len(sent) for para in batch['x'] for sent in para) == 0:
        #         new_JX = 1
        #     else:
        #         new_JX = max(len(sent) for para in batch['x'] for sent in para)
        #     JX = min(JX, new_JX)

        #     if sum(len(ques) for ques in batch['q']) == 0:
        #         new_JQ = 1
        #     else:
        #         new_JQ = max(len(ques) for ques in batch['q'])
        #     JQ = min(JQ, new_JQ)

        # if config.cpu_opt:
        #     if sum(len(para) for para in batch['x']) == 0:
        #         new_M = 1
        #     else:
        #         new_M = max(len(para) for para in batch['x'])
        #     M = min(M, new_M)

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
        
        # feed_dict[self.emb_mat] = emb_dict

        X = batch['x']
        CX = batch['cx']

        # if supervised:
        y = np.zeros([N, M, JX], dtype='bool')
        y2 = np.zeros([N, M, JX], dtype='bool')
        feed_dict[self.y] = y
        feed_dict[self.y2] = y2

        for i, yi in enumerate(batch['y']):    
            [j, k] = yi[0]
            [j2, k2] = yi[1]
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
                q[i, j] = _get_word(qij)
                q_mask[i, j] = True

        for i, cqi in enumerate(batch['cq']):
            for j, cqij in enumerate(cqi):
                for k, cqijk in enumerate(cqij):
                    cq[i, j, k] = _get_char(cqijk)
                    if k + 1 == config.max_word_size:
                        break

        return feed_dict

def bi_attention(config, is_train, h, u, h_mask=None, u_mask=None, scope=None, tensor_dict=None):
    with tf.variable_scope(scope or "bi_attention"):
        JX = tf.shape(h)[2]
        M = tf.shape(h)[1]
        JQ = tf.shape(u)[1]
        h_aug = tf.tile(tf.expand_dims(h, 3), [1, 1, 1, JQ, 1])
        u_aug = tf.tile(tf.expand_dims(tf.expand_dims(u, 1), 1), [1, M, JX, 1, 1])
        if h_mask is None:
            hu_mask = None
        else:
            h_mask_aug = tf.tile(tf.expand_dims(h_mask, 3), [1, 1, 1, JQ])
            u_mask_aug = tf.tile(tf.expand_dims(tf.expand_dims(u_mask, 1), 1), [1, M, JX, 1])
            hu_mask = h_mask_aug & u_mask_aug

        u_logits = get_logits([h_aug, u_aug], None, True, wd=config.wd, mask=hu_mask,
                              is_train=is_train, func=config.logit_func, scope='u_logits')  # [N, M, JX, JQ]
        u_a = softsel(u_aug, u_logits)  # [N, M, JX, d]
        h_a = softsel(h, tf.reduce_max(u_logits, 3))  # [N, M, d]
        h_a = tf.tile(tf.expand_dims(h_a, 2), [1, 1, JX, 1])

        if tensor_dict is not None:
            a_u = tf.nn.softmax(u_logits)  # [N, M, JX, JQ]
            a_h = tf.nn.softmax(tf.reduce_max(u_logits, 3))
            tensor_dict['a_u'] = a_u
            tensor_dict['a_h'] = a_h
            variables = tf.get_collection(tf.GraphKeys.VARIABLES, scope=tf.get_variable_scope().name)
            for var in variables:
                tensor_dict[var.name] = var

        return u_a, h_a


def attention_layer(config, is_train, h, u, h_mask=None, u_mask=None, scope=None, tensor_dict=None):
    with tf.variable_scope(scope or "attention_layer"):
        JX = tf.shape(h)[2]
        M = tf.shape(h)[1]
        JQ = tf.shape(u)[1]
        if config.q2c_att or config.c2q_att:
            u_a, h_a = bi_attention(config, is_train, h, u, h_mask=h_mask, u_mask=u_mask, tensor_dict=tensor_dict)
        if not config.c2q_att:
            u_a = tf.tile(tf.expand_dims(tf.expand_dims(tf.reduce_mean(u, 1), 1), 1), [1, M, JX, 1])
        if config.q2c_att:
            p0 = tf.concat([h, u_a, h * u_a, h * h_a], 3)
        else:
            p0 = tf.concat([h, u_a, h * u_a], 3)
        return p0
# if __name__ == "__main__":


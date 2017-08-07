import tensorflow as tf
import numpy
import tensorflow.contrib.rnn as rnn
import tensorflow.nn as nn
def get_cell_type(model):
    if model == 'rnn':
        cell_type = rnn.BasicRNNCell
    elif model == 'lstm':
        cell_type = rnn.BasicLSTMCell
    elif model == 'gru':
        cell_type = rnn.GRUCell
    else:
        raise Exception('model type not supported:{}'.format(model))
    return cell_type

def muliti_layers_init(rnn_size, layer_nums, cell_type):
    #every element stands for a layer
    cells = []
    for _ in range(layer_nums):
        cell = cell_type(rnn_size)
        cells.append(cell)
    return cells
def single_layer_init(rnn_size, cell_type):
    cell = cell_type(rnn_size)
    return cell 

def Rnn(extension=None, cell_fw, cell_bw=None, inputs, sequence_lengths=None, init_fw=None, init_bw=None, dtype=None):
    if extension == 'bi':
        return nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs, sequence_lengths, initial_state_fw=init_fw, initial_state_bw=init_bw dtype=dtype)
    elif extension == None:
        return nn.dynamic_rnn(cell_fw, inputs, sequence_lengths, initial_state=init_fw, dtype=dtype)
    else :
        raise Exception('Extension type error:{}'.format(extension))
# class Encoder:
#     def __init__(self, 
#                  model,
#                  rnn_size,
#                  layer_nums,
#                  input_dim,
#                  batch_size,
#                  dtype,
#                  inputs):
        
#         cell_type = get_cell_type(model)
#         #cells = muliti_layers_cells_init(rnn_size, layer_nums, cell_type)
        
#         #self.cell = cell = rnn.MultiRNNCell(cells)
#         self.cell = cell = cell_type(rnn_size)

#         #placeholder
#         self.input_holder  = tf.placeholder(dtype, [batch_size, input_dim])
#         self.target_holder = tf.placeholder(dtype, [batch_size, output_dim])
#         self.initial_state = cell.zero_state(batch_size, dtype)

#     def build_model():
#         #the output
#         self.output = Rnn('bi', cell, , inputs, sequence_lengths, self.initial_state,  , dtype=dtype)
#         return self.output

# class Interactor:
#     def __init__(self, args):

        
#         return 
# class Pointer:
#     def __init__(self, args):
#         return 

'''
parameters{
    batch_size,
    learning_rate,
    dropout_rate,
    dtype,
    lr
}
'''
class Model:
    def __init(self, batch_size, max_word_size, word_emb_size):

        # self.batch_size = batch_size
        # self.learning_rate = learning_rate
        # self.dropout_rate = dropout_rate
        # self.lr = lr
        self.config = config
        
        # x means the indexes of words of para in the emb_dict
        self.x = tf.placeholder('int32', [batch_size, None, None])
        self.cx = tf.placeholder('int32', [batch_size, None, None, max_word_size])
        self.q = tf.placeholder('int32', [batch_size, None])
        self.cq = tf.placeholder('int32', [batch_size, None, max_word_size])
        self.y = tf.placeholder('bool', [batch_size, None, None])
        self.y2 = tf.placeholder('bool', [batch_size, None, None])

        self.emb_mat = tf.placeholder('float', [None, word_emb_size])


        self.build_forward()
        
        self.build_loss()
        self.summary = tf.merge_all_summaries()
        self.summary = tf.merge_summary(tf.get_collection("summaries", scope=self.scope))
        # self.initial_state = 
        # self.loss = 
        #training method
        # self.optimizer = tf.train.AdamOptimizer(self.lr)

    def build_forward(self):

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

           
            with tf.variable_scope("emb_var"), tf.device("/cpu:0"):
                if config.mode == 'train':
                    word_emb_mat = tf.get_variable("word_emb_mat", dtype='float', shape=[VW, dw], initializer=get_initializer(config.emb_mat))
                else:
                    word_emb_mat = tf.get_variable("word_emb_mat", shape=[VW, dw], dtype='float')
                if config.use_glove_for_unk:
                    word_emb_mat = tf.concat(0, [word_emb_mat, self.new_emb_mat])

            with tf.name_scope("word"):
                Ax = tf.nn.embedding_lookup(word_emb_mat, self.x)  # [N, M, JX, d]
                Aq = tf.nn.embedding_lookup(word_emb_mat, self.q)  # [N, JQ, d]
                self.tensor_dict['x'] = Ax
                self.tensor_dict['q'] = Aq
            if config.use_char_emb:
                xx = tf.concat(3, [xx, Ax])  # [N, M, JX, di]
                qq = tf.concat(2, [qq, Aq])  # [N, JQ, di]
            else:
                xx = Ax
                qq = Aq

        with tf.variable_scope('encoding'):
            cell_type = get_cell_type(config.encoding_cell_type)
            cell = cell_type(config.hidden_size_encoding, state_is_tuple=True)
        with tf.variable_scope('interaction'):

        with tf.variable_scope('answer'):

    def get_feed_dict(self, batch):
        
if __name__ == "__main__":


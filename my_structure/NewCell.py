import tensorflow as tf
from tensorflow.python.ops.rnn_cell import RNNCell, GRUCell
from tensorflow.python.ops import math_ops
from utils.general import exp_mask, flatten
from tensorflow.python.ops.rnn_cell_impl import _linear
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import array_ops
class GeneGRUCell(RNNCell):

    def __init__(self,num_units, h, cont_size, activation=None,reuse=None,kernel_initializer=None,bias_initializer=None):

        super(GeneGRUCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._activation = activation or math_ops.tanh
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer
        self._gate_linear = None
        self._candidate_linear = None
        self.h = h
        self.cont_size = cont_size
    @property
    def state_size(self):
        return tuple([self._num_units, self.cont_size])

    @property
    def output_size(self):
        return self._num_units+self.cont_size

    def call(self, inputs, state):
       
        d, cont = state
        with tf.variable_scope(tf.get_variable_scope()):
            
            batch_size = self.h.get_shape().as_list()[0]
            seq_len    = tf.shape(self.h)[1]
            emb_size   = self.h.get_shape().as_list()[2]#tf.shape(self.h)[2]
            flat_h = tf.reshape(self.h, [-1, emb_size]) #flatten h    
            bs_times_seqlen = flat_h.get_shape().as_list()[0]#tf.shape(flat_h)[0]    
            tile_state = tf.tile(d, [seq_len, 1])
            
            '''2 linear layer should be seperated, because of the use of kernel'''
            with tf.variable_scope("val"):
                val = self._activation(_linear([tile_state, flat_h], self.state_size[0], True))
            with tf.variable_scope("s"):
                s = _linear([val], 1, True, bias_initializer=tf.constant_initializer(0))
                s = tf.reshape(s, [batch_size, -1]) #[batch_size, seq_len]
                a = tf.nn.softmax(s, 1) #[batch_size, seq_length]
                a = tf.reshape(a, [-1])
                flat_h = tf.transpose(flat_h)  
                cont = flat_h*a
                cont = tf.transpose(cont)
                cont = tf.reshape(cont, [batch_size, -1, emb_size])
                cont = tf.reduce_sum(cont, 1)  #[batch_size, emb_size]
                new_inputs = tf.concat([inputs, cont],1)

        '''u: z_t'''
        with tf.variable_scope("gates"):  # Reset gate and update gate.
            bias_ones = self._bias_initializer
            if self._bias_initializer is None:
                dtype = [a.dtype for a in [new_inputs, d]][0]
                bias_ones = init_ops.constant_initializer(1.0, dtype=dtype)
            value = math_ops.sigmoid(
                _linear([new_inputs, d], 2 * self._num_units, True, bias_ones,
                    self._kernel_initializer))
            r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)
        with tf.variable_scope("candidate"):
            c = self._activation(
                _linear([new_inputs, r * d], self._num_units, True,
                    self._bias_initializer, self._kernel_initializer))
        new_h = u * d + (1 - u) * c

        return tf.concat([new_h, cont],1), tuple([new_h,cont])


class GeneCell(RNNCell):
    def __init__(self, cell, h, input_keep_prob=1.0, is_train=None):
        self._cell = cell
        self.h = h
        # self.flat_h = flatten(h, )
        self._activation = math_ops.tanh
        # self._kernel_initializer = kernel_initializer
        # self._bias_initializer = bias_initializer
    @property
    def output_size(self):
        return self._cell.output_size

    @property
    def state_size(self):
        return self._cell.state_size

    def __call__(self, inputs, state, name=None):
    	
    	with tf.variable_scope(tf.get_variable_scope()):
            # print(inputs)
            batch_size = self.h.get_shape().as_list()[0]
            seq_len    = tf.shape(self.h)[1]
            emb_size   = self.h.get_shape().as_list()[2]#tf.shape(self.h)[2]
            flat_h = tf.reshape(self.h, [-1, emb_size]) #flatten h    
            bs_times_seqlen = flat_h.get_shape().as_list()[0]#tf.shape(flat_h)[0]    
            tile_state = tf.tile(state, [seq_len, 1])
          
            '''2 linear layer should be seperated, because of the use of kernel'''
            with tf.variable_scope("val"):
                val = self._activation(_linear([tile_state, flat_h], self.state_size, False))
            with tf.variable_scope("s"):
                s = _linear([val], 1, False)
                s = tf.reshape(s, [batch_size, -1]) #[batch_size, seq_len]
                a = tf.nn.softmax(s, 1) #[batch_size, seq_length]
                a = tf.reshape(a, [-1])
                flat_h = tf.transpose(flat_h)  
                cont = flat_h*a
                cont = tf.transpose(cont)
                cont = tf.reshape(cont, [batch_size, -1, emb_size])
                cont = tf.reduce_sum(cont, 1)  #[batch_size, emb_size]
                new_inputs = tf.concat([inputs, cont],1)
                print('success')
            return self._cell(new_inputs, state)

def get_maxout(batch_of_tensors):
    batch_size = batch_of_tensors.get_shape().as_list()[0]
    hid_size   = batch_of_tensors.get_shape().as_list()[2]
    half_hid_size   = (int)(hid_size/2)
    reshaped = tf.reshape(batch_of_tensors, [batch_size, -1, half_hid_size, 2])
    reshaped = tf.reduce_max(reshaped, 3)
    return reshaped
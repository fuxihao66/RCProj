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
class Encoder:
    def __init__(self, 
                 model,
                 rnn_size,
                 layer_nums,
                 input_dim,
                 batch_size,
                 dtype,
                 inputs):
        
        cell_type = get_cell_type(model)
        #cells = muliti_layers_cells_init(rnn_size, layer_nums, cell_type)
        
        #self.cell = cell = rnn.MultiRNNCell(cells)
        self.cell = cell = cell_type(rnn_size)

        #placeholder
        self.input_holder  = tf.placeholder(dtype, [batch_size, input_dim])
        self.target_holder = tf.placeholder(dtype, [batch_size, output_dim])
        self.initial_state = cell.zero_state(batch_size, dtype)

    def build_model():
        #the output
        self.output = Rnn('bi', cell, , inputs, sequence_lengths, self.initial_state,  , dtype=dtype)
        return self.output

class Interactor:
    def __init__(self, args):

        
        return 
class Pointer:
    def __init__(self, args):
        return 
class OurModel:
    def __init(self, 
               batch_size,
               learning_rate,
               dropout_rate,
               dtype,
               lr):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.lr = lr
        
    
        encoder_layer = Encoder('lstm', 100, 1, input_size, batch_size, dtype)
        interaction_layer = Interactor(encoder_layer.output)
        pointing_layer = Pointer()
        
        
        self.initial_state = 
        self.loss = 
        #training method
        self.optimizer = tf.train.AdamOptimizer(self.lr)



if __name__ == "__main__":


import math
from metadata_operation import *
import numpy as np
class DataSet:
    '''
    the data_dict looks like:
    {'xs':[x1_para, x2_para, ..],
     'cxs':[]
     'qs':[q1_sent, q2_sent, ..],
     'as':[]}
     x means the word level 
     cx means the char level
    '''
    def __init__(self, data_dict):

        self.data = data_dict
        self.num_examples = len(data_dict['queries'])
        # self.batches = []
        self.w2v_dict, self.w2i_dict = get_word2vec_from_file()
        self.embed_mat = []
        for key in self.w2v_dict:
            self.embed_mat.append(w2v_dict[key])
        self.embed_mat = np.asarray(self.embed_mat)

    def generate_batch(self, batch_size, shuffle=False):
        num_batch = int(math.ceil(self.num_examples/batch_size))     

        if shuffle:

        for _ in range(num_batch):
            for xs in self.data['passages']:
                
                self.batches.append(batch)
            for ans in self.data['answers']:

            for qus in self.data['queries']:

    '''
    ## the case should be considered when constructing the dict?????
    '''
    def get_word_idx_in_dict(self, word):
        for key in self.w2i_dict:
            if word == key:
                return w2i_dict[key]
            elif word == key.capitalize():
                return w2i_dict[key]
            elif word == key.upper():
            elif word == key.lower():

if __name__ == '__main__':

    train_data_dict = read_data_as_a_passage(path_to_train)
    dev_data_dict   = read_data_as_a_passage(path_to_dev)
    
    tokenized_train_data = {}
    tokenized_dev_data   = {}
    tokenized_train_data['char'] = []
    tokenized_dev_data['char'] = []
    for key in train_data_dict:
        tokenized_train_data[key] = Tokenize(train_data_dict[key])
        if key == 'passages':
            for passage in tokenized_train_data[key]:
                cxi = [[list(xijk) for xijk in xij] for xij in passage]
                tokenized_train_data['char'].append(cxi)
    for key in dev_data_dict:
        tokenized_dev_data[key]   = Tokenize(dev_data_dict[key])
        if key == 'passages':
            for passage in tokenized_dev_data[key]:
                cxi = [[list(xijk) for xijk in xij] for xij in passage]
                tokenized_dev_data['char'].append(cxi)
    #do some char level modification

    data_set_train = DataSet(tokenized_train_data)
    data_set_dev   = DataSet(tokenized_dev_data)

    batch_train = data_set_train.generate_batch()
    batch_dev   = data_set_dev.generate_batch()




    
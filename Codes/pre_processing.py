import math
from metadata_operation import *
import numpy as np
from tqdm import tqdm
from rouge_operation import *
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
    def __init__(self, data_dict, batch_size):

        self.batch_size = batch_size
        self.data = data_dict
        self.num_examples = len(data_dict['queries'])
        self.batches = []
        # self.w2v_dict, self.w2i_dict = get_word2vec_from_file()
        # self.embed_mat = []
        # for key in self.w2v_dict:
        #     self.embed_mat.append(w2v_dict[key])
        # self.embed_mat = np.asarray(self.embed_mat)
        
        self.operate_answers()
        self.tokenize() 
        self.generate_batch(self.batch_size, shuffle=False)

    def generate_batch(self, batch_size, shuffle=False):
        num_batch = int(math.ceil(self.num_examples/batch_size))     

        # if shuffle:


        for i in tqdm(range(self.num_batch)):
            batch = {}
            if (i+1)*batch_size <= self.num_examples:
                batch['x']   = self.data['passages'][i*batch_size:(i+1)*batch_size]
                batch['cx']  = self.data['char_p'][i*batch_size:(i+1)*batch_size]
                batch['y']   = self.data['ans_start_stop_idx'][i*batch_size:(i+1)*batch_size]
                batch['q']   = self.data['queries'][i*batch_size:(i+1)*batch_size]
                batch['cq']  = self.data['char_q'][i*batch_size:(i+1)*batch_size]
            else :
                batch['x']   = self.data['passages'][i*batch_size:num_examples]
                batch['cx']  = self.data['char_x'][i*batch_size:num_examples]
                batch['y']   = self.data['ans_start_stop_idx'][i*batch_size:num_examples]
                batch['q']   = self.data['queries'][i*batch_size:num_examples]
                batch['cq']  = self.data['char_q'][i*batch_size:num_examples]
            self.batches.append(batch)   
            print('batch data preparation finished')    
        return batches



    def get_batch_list(self):
        return batches

    def tokenize(self):
        self.data['char_x'] = []
        self.data['char_q'] = []
        for key in self.data:
            if key != 'answers':
                self.data[key] = Tokenize(self.data[key])
            if key == 'passages':
                for passage in self.data[key]:
                    cxi = [[list(xijk) for xijk in xij] for xij in passage]
                    self.data['char_x'].append(cxi)
            elif key == 'queries':
                for question in self.data[key]:
                    cqi = [list(qij) for qij in question]
                    self.data['char_q'].append(cqi)


    def operate_answers(self):
        self.data['ans_start_stop_idx'] = []

        for i in range(len(self.data['passages'])):
            para = self.data['passages'][i]
            # ans  = del_signal(self.data['answers'][i])
            ans = self.data['answers'][i]
            l, flag = get_highest_rl_span(para, ans, 30)
            if  flag == False:
                l = get_selected_span(para, data['passage_selected'][0])
                # l looks like: [[j1,k1],[j2,k2]]
            self.data['ans_start_stop_idx'].append(l)
    '''
    ## the case should be considered when constructing the dict?????
    '''
    # def get_word_idx_in_dict(self, word):
    #     for key in self.w2i_dict:
    #         if word == key:
    #             return w2i_dict[key]
    #         elif word == key.capitalize():
    #             return w2i_dict[key]
    #         elif word == key.upper():
    #         elif word == key.lower():

if __name__ == '__main__':
    train_data_dict = read_data_as_a_passage(path_to_train)
    dev_data_dict   = read_data_as_a_passage(path_to_dev)

    train_data = DataSet(train_data_dict, config.batch_size)
    dev_data   = DataSet(dev_data_dict, config.batch_size)





    
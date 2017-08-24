import math
from metadata_operation import *
import numpy as np
from tqdm import tqdm
from rouge_operation import *
from multiprocessing import Process
from multiprocessing import Queue
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
        self.data['ans_start_stop_idx'] = []
        # self.temp = []
        self.num_examples = len(data_dict['passages'])
        self.batches = []
        
    def generate_batch(self, batch_size, shuffle=False):
        num_batch = int(math.ceil(self.num_examples/batch_size))     

        # if shuffle:
        for i in tqdm(range(num_batch)):
            batch = {}
            if (i+1)*batch_size <= self.num_examples:
                batch['x']   = self.data['passages'][i*batch_size:(i+1)*batch_size]
                batch['cx']  = self.data['char_x'][i*batch_size:(i+1)*batch_size]
                batch['y']   = self.data['ans_start_stop_idx'][i*batch_size:(i+1)*batch_size]
                batch['q']   = self.data['queries'][i*batch_size:(i+1)*batch_size]
                batch['cq']  = self.data['char_q'][i*batch_size:(i+1)*batch_size]
            else :
                batch['x']   = self.data['passages'][i*batch_size:self.num_examples]
                batch['cx']  = self.data['char_x'][i*batch_size:self.num_examples]
                batch['y']   = self.data['ans_start_stop_idx'][i*batch_size:self.num_examples]
                batch['q']   = self.data['queries'][i*batch_size:self.num_examples]
                batch['cq']  = self.data['char_q'][i*batch_size:self.num_examples]
            self.batches.append(batch)   
        print('batch data preparation finished')    
        



    def get_batch_list(self):
        return self.batches

    def tokenize(self):
        self.data['char_x'] = []
        self.data['char_q'] = []
        for key in self.data:
            print(key)
            if key == 'passages':
                self.data[key] = Tokenize(self.data[key])
                for passage in self.data[key]:
                    cxi = [[list(xijk) for xijk in xij] for xij in passage]
                    self.data['char_x'].append(cxi)
            elif key == 'queries':
                self.data[key] = Tokenize_without_sent(self.data[key])
                for question in self.data[key]:
                    cqi = [list(qij) for qij in question]
                    self.data['char_q'].append(cqi)
    
    def operate_answers_single_thread(self, start, end, q):
        temp = []
        for i in range(end)[start:]:
            para = self.data['passages'][i]         
            # ans  = del_signal(self.data['answers'][i])
            ans = self.data['answers'][i]
            l, flag = get_highest_rl_span(para, ans, 40)
            if  flag == False:
                l = get_selected_span(para, self.data['passage_selected'][i][0])
                # l looks like: [[j1,k1],[j2,k2]]
            # self.data['ans_start_stop_idx'].append(l)
            temp.append(l)

        q.put(temp)
        
    def operate_answers(self, num_threads):
        
        # word_dict, _, __ = get_word2idx_and_embmat('''/home/zhangs/RC/data/glove.6B.100d.txt''') 

        # def del_signal(sentence):
        #     token_sent = Tokenize_string_word_level(sentence)
        #     flag = 0
        #     for word in word_dict:
        #         if token_sent[0] in (word, word.lower(), word.capitalize(), word.upper()):
        #             flag = 1
        #             break
        #     if flag == 0:
        #         sentence[0] = sentence[0].lower()
        #     return sentence[:len(sentence)-1]
        
        each_size = int(math.ceil(self.num_examples/num_threads)) 
        thread_list = []
        q = Queue()

        for thread_idx in tqdm(range(num_threads)):
            # self.temp.append([])
            if thread_idx == (num_threads-1):
                thread_list.append(Process(target=self.operate_answers_single_thread, args=(thread_idx*each_size,len(self.data['passages']),q)))
            else:
                thread_list.append(Process(target=self.operate_answers_single_thread, args=(thread_idx*each_size, (thread_idx+1)*each_size,q)))
            
        for thr in thread_list:
            print('thread start')
            thr.start()
        for thr in thread_list:
            thr.join()

        while  q.qsize() > 0:  
            l = q.get()
            for item in l:
                self.data['ans_start_stop_idx'].append(item)

        print(self.data['ans_start_stop_idx'])
        # for i in tqdm(range(len(self.data['passages']))):
          
    def write_answers_to_file(self, path):
        with open(path, 'w', encoding='utf8') as data_file:
            data_file.write(json.dumps(self.data['ans_start_stop_idx']))
        
    def read_operated_answers_from_file(self, path):
        with open(path, 'r', encoding='utf8') as data_file:
            for line in tqdm(data_file):
                instance = json.loads(line)
                self.data['ans_start_stop_idx'] = instance[0:300]

    def init_with_ans_file(self, path_to_answers, batch_size):
        self.read_operated_answers_from_file(path_to_answers)
        self.tokenize()
        self.generate_batch(batch_size)


if __name__ == '__main__':
    train_data_dict = read_metadata('''/home/zhangs/RC/data/train_v1.1.json''')
    # dev_data_dict   = read_metadata('''/home/zhangs/RC/data/dev_v1.1.json''')

    train_data = DataSet(train_data_dict)
    # dev_data   = DataSet(dev_data_dict)
    print('start operating answers')
    train_data.operate_answers(26)
    print('operating answers successfully')
    train_data.write_answers_to_file('''/home/zhangs/RC/data/train_answers.json''')
    # print('start operating answers')
    # dev_data.operate_answers(20)
    # print('operating answers successfully')
    # dev_data.write_answers_to_file('''/home/zhangs/RC/data/dev_answers.json''')



    
# -*- coding:utf8 -*-
import codecs
import json
import sys

def read_data_as_a_passage(file_to_read, file_to_write):
    passage_list     =  []
    answers_list     =  []
    query_list       =  []
    description_list =  []
    with open(file_to_read, 'r', encoding='utf8') as data_file:
        for line in data_file:
            instance = json.loads(line)
            passage = ''
            for sentence in instance['passages']:
                passage = passage + sentence['passage_text']
            passage_list.append(passage)

            answer = ''
            for answer_str in instance['answers']:
                answer = answer + answer_str
            
            answers_list.append(answer) 
            #some answers are blank???????????????

            query_list.append(instance['query'])
            description_list.append(instance['query_type'])

    data_to_write = {}
    # data_to_write['passages']     =  passage_list
    # data_to_write['answers']      =  answers_list
    # data_to_write['queries']      =  query_list
    # data_to_write['descriptions'] =  description_list
    


    # ### this method cannot encode some character very well
    # with open(file_to_write, 'w') as modified_data_file:
    #     modified_data_file.write(json.dumps(data_to_write)) 

    
if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('parameters error')
    else:
        read_data_as_a_passage(sys.argv[1], sys.argv[2])
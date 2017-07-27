import json

def read_data_different_passages(file_path):


### what to do with the "is_selected" tag????
def read_data_as_a_passage(file_path):
    passage_list     =  []
    answers_list     =  []
    query_list       =  []
    description_list =  []
    with open(file_path, 'r') as data_file:
        data_in_list = json.loads(data_file)
        for instance in data_in_list:
            passage = ''
            for sentence in instance['passages']:
                passage = passage + sentence['passage_text']
            passage_list.append(passage)

            answers_list.append(instance['answers']) #may cause exception

            query_list.append(instance['query'])
            description_list.append(instance['query_type'])
        
    return data_set 
def load_file():
    if len(sys.argv) != 2:
        print('Parameter error. The pattern is like \"python3 test.py data.json \"')
    else:
        read_data()
    return
if __name__ == "__main__":
    load_file()
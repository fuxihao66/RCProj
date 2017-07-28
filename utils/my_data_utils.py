# class DataSet:
#     def __init__(self, data):
#         self.data = data
#         self.total_num = self.get_data_size()

#     def get_data_size(self):

'''
1. how to deal with answers, passages and queries
2. what kind of data structure
'''

def read_from_file(file_type):
    if file_type == 'train':

    elif file_type == 'dev':
    
    return dict_of_data
def read_data_from_dict(batch_size, data):
    exampels_num = len(data)
    batch_num = exampels_num/batch_size
    data_set = []
    for _ in range(batch_num):

    return data_set
def get_batch(data_set, idx):
    return data_set[idx]

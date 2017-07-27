class DataSet:
    def __init__(self, data):
        self.data = data
        self.total_num = self.get_data_size()

    def get_data_size(self):
        
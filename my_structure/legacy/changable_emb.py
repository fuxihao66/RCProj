
word_emb = tf.get_variable("word_emb", 
                            shape=[config.word_vocab_size, config.word_emb_size], 
                            dtype='float', 
                            initializer=self.emb_mat)

def get_word2idx_and_embmat(path_to_file):
    word2vec_dict = {}
    word2idx_dict = {}
    i = 0
    with open(path_to_file, 'r') as vec_file:
        for line in tqdm(vec_file):
            list_of_line = line.split(' ')
            word2vec_dict[list_of_line[0]] = list(map(float, list_of_line[1:]))
            i += 1
            word2idx_dict[list_of_line[0]] = i
    emb_mat = []
    emb_mat.append([0 for i in range(100)])
    for key in word2vec_dict:
        emb_mat.append(word2vec_dict[key])
    emb_mat = np.asarray(emb_mat)
    emb_mat = emb_mat.astype(dtype='float32')
    vacabulary_size = i
    return word2idx_dict, emb_mat, vacabulary_size

def get_feed_dict(self):
    if supervised:
        y = np.zeros([N, ans_length, voca_size], dtype='float') 
        feed_dict[self.y] = y

        '''batch_y: [batch_size, ans_len]'''
        for i, yi in enumerate(batch['y']):  
            for j, yij in enumerate(yi):
                k = _get_word(yij) 
                y[i, j, k] = 1.0


self.y = tf.placeholder('float', [config.batch_size, config.max_ans_length, config.word_vocabulary_size])
#[batch_size, ans_length, vocabulary_size] 
# like [ [ [0,0,..,1.,..,0],[1,...,0] ],... ]
from tensorflow.python.ops.losses.losses_impl import *
def build_loss(self):
    loss = tf.losses.log_loss(labels=self.y, predictions=self.y0, reduction=Reduction.NONE)
    loss = loss*self.y
    loss = tf.reduce_sum(loss, 2)
    loss = tf.reduce_sum(loss, 1)
    loss = tf.reduce_mean(loss, 0)
    
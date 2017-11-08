import re
import nltk
from tqdm import tqdm

def word_tokenize(tokens):
    return [token.replace("''", '"').replace("``", '"') for token in nltk.word_tokenize(tokens)]
def process_tokens(temp_tokens):
    tokens = []
    for token in temp_tokens:
        flag = False
        l = ("-", "\u2212", "\u2014", "\u2013", "/", "~", '"', "'", "\u201C", "\u2019", "\u201D", "\u2018", "\u00B0")
        # \u2013 is en-dash. Used for number to nubmer
        # l = ("-", "\u2212", "\u2014", "\u2013")
        # l = ("\u2013",)
        tokens.extend(re.split("([{}])".format("".join(l)), token))
    return tokens


''' def get_word2vec(args, word_counter):
    glove_path = os.path.join(args.glove_dir, "glove.{}.{}d.txt".format(args.glove_corpus, args.glove_vec_size))
    sizes = {'6B': int(4e5), '42B': int(1.9e6), '840B': int(2.2e6), '2B': int(1.2e6)}
    total = sizes[args.glove_corpus]
    word2vec_dict = {}
    with open(glove_path, 'r', encoding='utf-8') as fh:
        for line in tqdm(fh, total=total):
            array = line.lstrip().rstrip().split(" ")
            word = array[0]
            vector = list(map(float, array[1:]))
            if word in word_counter:
                word2vec_dict[word] = vector
            elif word.capitalize() in word_counter:
                word2vec_dict[word.capitalize()] = vector
            elif word.lower() in word_counter:
                word2vec_dict[word.lower()] = vector
            elif word.upper() in word_counter:
                word2vec_dict[word.upper()] = vector

    print("{}/{} of word vocab have corresponding vectors in {}".format(len(word2vec_dict), len(word_counter), glove_path))
    return word2vec_dict '''

def get_word2vec_from_file(path_to_file):
    word2vec_dict = {}
    word2idx_dict = {}
    i = 0
    with open(path_to_file, 'r') as vec_file:
        for line in tqdm(vec_file):
            list_of_line = line.split(' ')
            word2vec_dict[list_of_line[0]] = list(map(float, list_of_line[1:]))
            word2idx_dict[list_of_line[0]] = i
            i = i+1
    return word2vec_dict, word2idx_dict

if __name__ == '__main__':
    sent_tokenize = nltk.sent_tokenize
    context = 'aklsdjfi. vncxz, mfiweofucsdopmv. zxm pokfzsl jkfweoprfkqwxzl foasp ifwkfznl  weuri owr us oiwe kasljas lksfjalfj kjf alsfjwiofpoaksdf;asmfa.sdmf.,asdmfnaoierjhiojdsfkalsdjfgasjsnajknfskdjfhasdf'
    xi = list(map(word_tokenize, sent_tokenize(context)))
    # xi = [process_tokens(tokens) for tokens in xi]
    cxi = [[list(xijk) for xijk in xij] for xij in xi]
    print(cxi)

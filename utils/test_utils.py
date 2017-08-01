import re
import nltk


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

if __name__ == '__main__':
    sent_tokenize = nltk.sent_tokenize
    context = ['aklsdjfiowenm,.vncxz,mfiweofucsdopmv.,zxm pokfzsl jkfweoprfkqwxzl  foasp ifwkfznl  weuri owr us oiwe kasljas lksfjalfj kjf alsfjwiofpoaksdf;asmfa.sdmf.,asdmfnaoierjhiojdsfkalsdjfgasjsnajknfskdjfhasdf','asdkfjkasldjfklasdfj']
    xi = list(map(word_tokenize, sent_tokenize(context)))
    xi = [process_tokens(tokens) for tokens in xi]
    cxi = [[list(xijk) for xijk in xij] for xij in xi]
    print(cxi)
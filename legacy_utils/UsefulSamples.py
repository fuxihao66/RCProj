# class DataSet:
#     def __init__(self, data):
#         self.data = data
#         self.total_num = self.get_data_size()

#     def get_data_size(self):

'''
1. how to deal with answers, passages and queries
2. what kind of data structure
'''





if config.use_glove_for_unk:
    # create new word2idx and word2vec
    word2vec_dict = shared['lower_word2vec'] if config.lower_word else shared['word2vec']
    new_word2idx_dict = {word: idx for idx, word in enumerate(word for word in word2vec_dict.keys() if word not in shared['word2idx'])}
    shared['new_word2idx'] = new_word2idx_dict
    offset = len(shared['word2idx'])
    word2vec_dict = shared['lower_word2vec'] if config.lower_word else shared['word2vec']
    new_word2idx_dict = shared['new_word2idx']
    idx2vec_dict = {idx: word2vec_dict[word] for word, idx in new_word2idx_dict.items()}
    # print("{}/{} unique words have corresponding glove vectors.".format(len(idx2vec_dict), len(word2idx_dict)))
    new_emb_mat = np.array([idx2vec_dict[idx] for idx in range(len(idx2vec_dict))], dtype='float32')
    shared['new_emb_mat'] = new_emb_mat


def prepro_each(args, data_type, start_ratio=0.0, stop_ratio=1.0, out_name="default", in_path=None):
    if args.tokenizer == "PTB":
        import nltk
        sent_tokenize = nltk.sent_tokenize
        def word_tokenize(tokens):
            return [token.replace("''", '"').replace("``", '"') for token in nltk.word_tokenize(tokens)]
    elif args.tokenizer == 'Stanford':
        from my.corenlp_interface import CoreNLPInterface
        interface = CoreNLPInterface(args.url, args.port)
        sent_tokenize = interface.split_doc
        word_tokenize = interface.split_sent
    else:
        raise Exception()

    if not args.split:
        sent_tokenize = lambda para: [para]

    source_path = in_path or os.path.join(args.source_dir, "{}-v1.1.json".format(data_type))
    source_data = json.load(open(source_path, 'r'))

    q, cq, y, rx, rcx, ids, idxs = [], [], [], [], [], [], []
    cy = []
    x, cx = [], []
    answerss = []
    p = []
    word_counter, char_counter, lower_word_counter = Counter(), Counter(), Counter()
    start_ai = int(round(len(source_data['data']) * start_ratio))
    stop_ai = int(round(len(source_data['data']) * stop_ratio))
    for ai, article in enumerate(tqdm(source_data['data'][start_ai:stop_ai])):
        xp, cxp = [], []
        pp = []
        x.append(xp)
        cx.append(cxp)
        p.append(pp)
        for pi, para in enumerate(article['paragraphs']):
            # wordss
            context = para['context']
            context = context.replace("''", '" ')
            context = context.replace("``", '" ')
            xi = list(map(word_tokenize, sent_tokenize(context)))
            xi = [process_tokens(tokens) for tokens in xi]  # process tokens
            # given xi, add chars
            cxi = [[list(xijk) for xijk in xij] for xij in xi]
            xp.append(xi)
            cxp.append(cxi)
            pp.append(context)

            for xij in xi:
                for xijk in xij:
                    word_counter[xijk] += len(para['qas'])
                    lower_word_counter[xijk.lower()] += len(para['qas'])
                    for xijkl in xijk:
                        char_counter[xijkl] += len(para['qas'])

            rxi = [ai, pi]
            assert len(x) - 1 == ai
            assert len(x[ai]) - 1 == pi
            for qa in para['qas']:
                # get words
                qi = word_tokenize(qa['question'])
                cqi = [list(qij) for qij in qi]
                yi = []
                cyi = []
                answers = []
                for answer in qa['answers']:
                    answer_text = answer['text']
                    answers.append(answer_text)
                    answer_start = answer['answer_start']
                    answer_stop = answer_start + len(answer_text)
                    # TODO : put some function that gives word_start, word_stop here
                    yi0, yi1 = get_word_span(context, xi, answer_start, answer_stop)
                    # yi0 = answer['answer_word_start'] or [0, 0]
                    # yi1 = answer['answer_word_stop'] or [0, 1]
                    assert len(xi[yi0[0]]) > yi0[1]
                    assert len(xi[yi1[0]]) >= yi1[1]
                    w0 = xi[yi0[0]][yi0[1]]
                    w1 = xi[yi1[0]][yi1[1]-1]
                    i0 = get_word_idx(context, xi, yi0)
                    i1 = get_word_idx(context, xi, (yi1[0], yi1[1]-1))
                    cyi0 = answer_start - i0
                    cyi1 = answer_stop - i1 - 1
                    # print(answer_text, w0[cyi0:], w1[:cyi1+1])
                    assert answer_text[0] == w0[cyi0], (answer_text, w0, cyi0)
                    assert answer_text[-1] == w1[cyi1]
                    assert cyi0 < 32, (answer_text, w0)
                    assert cyi1 < 32, (answer_text, w1)

                    yi.append([yi0, yi1])
                    cyi.append([cyi0, cyi1])

                for qij in qi:
                    word_counter[qij] += 1
                    lower_word_counter[qij.lower()] += 1
                    for qijk in qij:
                        char_counter[qijk] += 1

                q.append(qi)
                cq.append(cqi)
                y.append(yi)
                cy.append(cyi)
                rx.append(rxi)
                rcx.append(rxi)
                ids.append(qa['id'])
                idxs.append(len(idxs))
                answerss.append(answers)

            if args.debug:
                break

    word2vec_dict = get_word2vec(args, word_counter)
    lower_word2vec_dict = get_word2vec(args, lower_word_counter)

    # add context here
    data = {'q': q, 'cq': cq, 'y': y, '*x': rx, '*cx': rcx, 'cy': cy,
            'idxs': idxs, 'ids': ids, 'answerss': answerss, '*p': rx}
    shared = {'x': x, 'cx': cx, 'p': p,
              'word_counter': word_counter, 'char_counter': char_counter, 'lower_word_counter': lower_word_counter,
              'word2vec': word2vec_dict, 'lower_word2vec': lower_word2vec_dict}

    print("saving ...")
    save(args, data, shared, out_name)





def get_feed_dict(self, batch, is_train, supervised=True):
    assert isinstance(batch, DataSet)
    config = self.config
    N, M, JX, JQ, VW, VC, d, W = \
        config.batch_size, config.max_num_sents, config.max_sent_size, \
        config.max_ques_size, config.word_vocab_size, config.char_vocab_size, config.hidden_size, config.max_word_size
    feed_dict = {}

    if config.len_opt:
        """
        Note that this optimization results in variable GPU RAM usage (i.e. can cause OOM in the middle of training.)
        First test without len_opt and make sure no OOM, and use len_opt
        """
        if sum(len(sent) for para in batch.data['x'] for sent in para) == 0:
            new_JX = 1
        else:
            new_JX = max(len(sent) for para in batch.data['x'] for sent in para)
        JX = min(JX, new_JX)

        if sum(len(ques) for ques in batch.data['q']) == 0:
            new_JQ = 1
        else:
            new_JQ = max(len(ques) for ques in batch.data['q'])
        JQ = min(JQ, new_JQ)

    if config.cpu_opt:
        if sum(len(para) for para in batch.data['x']) == 0:
            new_M = 1
        else:
            new_M = max(len(para) for para in batch.data['x'])
        M = min(M, new_M)

    x = np.zeros([N, M, JX], dtype='int32')
    cx = np.zeros([N, M, JX, W], dtype='int32')
    x_mask = np.zeros([N, M, JX], dtype='bool')
    q = np.zeros([N, JQ], dtype='int32')
    cq = np.zeros([N, JQ, W], dtype='int32')
    q_mask = np.zeros([N, JQ], dtype='bool')

    feed_dict[self.x] = x
    feed_dict[self.x_mask] = x_mask
    feed_dict[self.cx] = cx
    feed_dict[self.q] = q
    feed_dict[self.cq] = cq
    feed_dict[self.q_mask] = q_mask
    feed_dict[self.is_train] = is_train
    if config.use_glove_for_unk:
        feed_dict[self.new_emb_mat] = batch.shared['new_emb_mat']

    X = batch.data['x']
    CX = batch.data['cx']

    if supervised:
         

        for i, (xi, cxi, yi) in enumerate(zip(X, CX, batch.data['y'])):
            start_idx, stop_idx = random.choice(yi)
            j, k = start_idx
            j2, k2 = stop_idx
            if config.single:
                X[i] = [xi[j]]
                CX[i] = [cxi[j]]
                j, j2 = 0, 0
            if config.squash:
                offset = sum(map(len, xi[:j]))
                j, k = 0, k + offset
                offset = sum(map(len, xi[:j2]))
                j2, k2 = 0, k2 + offset
            y[i, j, k] = True
            y2[i, j2, k2-1] = True

    def _get_word(word):
        d = batch.shared['word2idx']
        for each in (word, word.lower(), word.capitalize(), word.upper()):
            if each in d:
                return d[each]
        if config.use_glove_for_unk:
            d2 = batch.shared['new_word2idx']
            for each in (word, word.lower(), word.capitalize(), word.upper()):
                if each in d2:
                    return d2[each] + len(d)
        return 1

    def _get_char(char):
        d = batch.shared['char2idx']
        if char in d:
            return d[char]
        return 1

    for i, xi in enumerate(X):
        if self.config.squash:
            xi = [list(itertools.chain(*xi))]
        for j, xij in enumerate(xi):
            if j == config.max_num_sents:
                break
            for k, xijk in enumerate(xij):
                if k == config.max_sent_size:
                    break
                each = _get_word(xijk)
                assert isinstance(each, int), each
                x[i, j, k] = each
                x_mask[i, j, k] = True

    for i, cxi in enumerate(CX):
        if self.config.squash:
            cxi = [list(itertools.chain(*cxi))]
        for j, cxij in enumerate(cxi):
            if j == config.max_num_sents:
                break
            for k, cxijk in enumerate(cxij):
                if k == config.max_sent_size:
                    break
                for l, cxijkl in enumerate(cxijk):
                    if l == config.max_word_size:
                        break
                    cx[i, j, k, l] = _get_char(cxijkl)

    for i, qi in enumerate(batch.data['q']):
        for j, qij in enumerate(qi):
            q[i, j] = _get_word(qij)
            q_mask[i, j] = True

    for i, cqi in enumerate(batch.data['cq']):
        for j, cqij in enumerate(cqi):
            for k, cqijk in enumerate(cqij):
                cq[i, j, k] = _get_char(cqijk)
                if k + 1 == config.max_word_size:
                    break

    return feed_dict
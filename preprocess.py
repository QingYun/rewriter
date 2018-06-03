''' Handling the data io '''
import argparse
import torch
import transformer.Constants as Constants
import numpy as np
from nltk.tokenize import word_tokenize

def read_instances_from_file(inst_file, max_sent_len, keep_case):
    ''' Convert file into word seq lists and vocab '''

    word_insts = []
    trimmed_sent_count = 0
    with open(inst_file) as f:
        for sent in f:
            if not keep_case:
                sent = sent.lower()
            words = word_tokenize(sent)
            if len(words) > max_sent_len:
                trimmed_sent_count += 1
                word_insts += [None]
                continue

            word_inst = words[:max_sent_len]

            if word_inst:
                word_insts += [[Constants.BOS_WORD] + word_inst + [Constants.EOS_WORD]]
            else:
                word_insts += [None]

    print('[Info] Get {} instances from {}'.format(len(word_insts), inst_file))

    if trimmed_sent_count > 0:
        print('[Warning] {} instances are trimmed to the max sentence length {}.'
              .format(trimmed_sent_count, max_sent_len))

    return word_insts

def build_vocab_idx(word_insts, min_word_count):
    ''' Trim vocab by number of occurence '''

    full_vocab = set(w for sent in word_insts for w in sent)
    print('[Info] Original Vocabulary size =', len(full_vocab))

    word2idx = {
        Constants.BOS_WORD: Constants.BOS,
        Constants.EOS_WORD: Constants.EOS,
        Constants.PAD_WORD: Constants.PAD,
        Constants.UNK_WORD: Constants.UNK}

    word_count = {w: 0 for w in full_vocab}

    for sent in word_insts:
        for word in sent:
            word_count[word] += 1

    ignored_word_count = 0
    for word, count in word_count.items():
        if word not in word2idx:
            if count > min_word_count:
                word2idx[word] = len(word2idx)
            else:
                ignored_word_count += 1

    print('[Info] Trimmed vocabulary size = {},'.format(len(word2idx)),
          'each with minimum occurrence = {}'.format(min_word_count))
    print("[Info] Ignored word count = {}".format(ignored_word_count))
    return word2idx

def convert_instance_to_idx_seq(word_insts, word2idx):
    '''Word mapping to idx'''
    return [[word2idx[w] if w in word2idx else Constants.UNK for w in s] for s in word_insts]

def load_glove(path):
    embeddings = {}
    with open(path, 'r') as f:
        for l in f:
            line = l.split()
            word = line[0]
            vec = line[1:]
            embeddings[word] = np.array(vec)
    return embeddings

def load_structural_embeddings(path):
    return np.load(open(path, 'rb'))

def build_emb(embeddings, word2idx, semb):
    emb_dim = embeddings['a'].shape[0]
    semb_dim = semb.shape[1]
    emb = np.zeros((len(word2idx), emb_dim + semb_dim))
    empty_semb = np.zeros((semb_dim))
    idx2word = {v: k for k, v in word2idx.items()}
    for i in range(len(idx2word)):
        parts = idx2word[i].split('__')
        if len(parts) == 1:
            word = parts[0]
            sidx = None
        elif len(parts) == 2:
            word, sidx = parts
            try:
                sidx = int(sidx)
            except:
                sidx = None
        else:
            print('Error parts length:', idx2word[i])
            word = Constants.UNK_WORD
            sidx = None

        if word in embeddings:
            vec = embeddings[word]
        else:
            vec = np.random.normal(scale=0.6, size=(emb_dim, ))

        if sidx is not None:
            vec = np.append(vec, semb[sidx, :])
        else:
            vec = np.append(vec, empty_semb)

        emb[i] = vec
    return emb

def main():
    ''' Main function '''

    parser = argparse.ArgumentParser()
    parser.add_argument('-train_src', required=True)
    parser.add_argument('-train_tgt', required=True)
    parser.add_argument('-save_data', required=True)
    parser.add_argument('-glove', required=True)
    parser.add_argument('-semb', required=True)
    parser.add_argument('-max_len', '--max_word_seq_len', type=int, default=50)
    parser.add_argument('-min_word_count', type=int, default=5)
    parser.add_argument('-keep_case', action='store_true')
    parser.add_argument('-share_vocab', action='store_true')
    parser.add_argument('-vocab', default=None)

    opt = parser.parse_args()
    opt.max_token_seq_len = opt.max_word_seq_len + 2 # include the <s> and </s>

    # Training set
    train_src_word_insts = read_instances_from_file(
        opt.train_src, opt.max_word_seq_len, opt.keep_case)
    train_tgt_word_insts = read_instances_from_file(
        opt.train_tgt, opt.max_word_seq_len, opt.keep_case)

    if len(train_src_word_insts) != len(train_tgt_word_insts):
        print('[Warning] The training instance count is not equal.')
        min_inst_count = min(len(train_src_word_insts), len(train_tgt_word_insts))
        train_src_word_insts = train_src_word_insts[:min_inst_count]
        train_tgt_word_insts = train_tgt_word_insts[:min_inst_count]

    #- Remove empty instances
    train_src_word_insts, train_tgt_word_insts = list(zip(*[
        (s, t) for s, t in zip(train_src_word_insts, train_tgt_word_insts) if s and t]))

    # Build vocabulary
    if opt.vocab:
        predefined_data = torch.load(opt.vocab)
        assert 'dict' in predefined_data

        print('[Info] Pre-defined vocabulary found.')
        src_word2idx = predefined_data['dict']['src']
        tgt_word2idx = predefined_data['dict']['tgt']
    else:
        if opt.share_vocab:
            print('[Info] Build shared vocabulary for source and target.')
            word2idx = build_vocab_idx(
                train_src_word_insts + train_tgt_word_insts, opt.min_word_count)
            src_word2idx = tgt_word2idx = word2idx
        else:
            print('[Info] Build vocabulary for source.')
            src_word2idx = build_vocab_idx(train_src_word_insts, opt.min_word_count)
            print('[Info] Build vocabulary for target.')
            tgt_word2idx = build_vocab_idx(train_tgt_word_insts, opt.min_word_count)

    # index to embedding
    print('[Info] Link embedding')
    embeddings = load_glove(opt.glove)
    semb = load_structural_embeddings(opt.semb)
    src_emb = build_emb(embeddings, src_word2idx, semb)
    tgt_emb = build_emb(embeddings, tgt_word2idx, semb)

    print(embeddings['the'])
    print(tgt_emb[tgt_word2idx['the']])

    # word to index
    print('[Info] Convert source word instances into sequences of word index.')
    train_src_insts = convert_instance_to_idx_seq(train_src_word_insts, src_word2idx)

    print('[Info] Convert target word instances into sequences of word index.')
    train_tgt_insts = convert_instance_to_idx_seq(train_tgt_word_insts, tgt_word2idx)

    assert len(train_src_insts) == len(train_tgt_insts)
    total_sent = len(train_src_insts)
    valid_sent_idx = np.random.choice(total_sent, int(total_sent / 50))
    valid_src_insts = []
    valid_tgt_insts = []
    for idx in valid_sent_idx:
        valid_src_insts.append(train_src_insts[idx])
        valid_tgt_insts.append(train_tgt_insts[idx])
    for idx in sorted(valid_sent_idx, reverse=True):
        del train_src_insts[idx]
        del train_tgt_insts[idx]

    assert (len(train_src_insts) + len(valid_src_insts)) == total_sent

    print('src_word2idx', len(src_word2idx))
    print('tgt_word2idx', len(tgt_word2idx))
    print('src_emb', src_emb.shape)
    print('tgt_emb', tgt_emb.shape)
    print('train_src_insts', len(train_src_insts))
    print('train_tgt_insts', len(train_tgt_insts))
    print('valid_src_insts', len(valid_src_insts))
    print('valid_tgt_insts', len(valid_tgt_insts))

    data = {
        'settings': opt,
        'dict': {
            'src': src_word2idx,
            'tgt': tgt_word2idx},
        'emb': {
            'src': src_emb,
            'tgt': tgt_emb},
        'train': {
            'src': train_src_insts,
            'tgt': train_tgt_insts},
        'valid': {
            'src': valid_src_insts,
            'tgt': valid_tgt_insts}}

    print('[Info] Dumping the processed data to pickle file', opt.save_data)
    torch.save(data, opt.save_data)
    print('[Info] Finish.')

if __name__ == '__main__':
    main()

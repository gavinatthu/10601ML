import numpy as np
import sys

def data_reading(dir_in):
    sentences = []
    with open(dir_in,'r') as inf:
        lines = inf.readlines()
        sequences = []
        sequence = []
        for line in lines:
            if line == '\n':
                sequences.append(sequence)
                sequence = []
            else:
                sequence.append(line.strip('\n').split('\t') )
    return sequences

def index_reading(idx_in):
    with open(idx_in,'r') as inf:
        lines = inf.readlines()
        item2idx = {x.strip('\n'):idx for idx, x in enumerate(lines)}
    return item2idx

def hmm(sequences, w2i, t2i):
    pi = np.ones(len(t2i.keys()))
    trans = np.ones((len(t2i.keys()), len(t2i.keys())))
    emit = np.ones((len(t2i.keys()), len(w2i.keys())))
    for sequence in sequences:
        for i, item in enumerate(sequence):
            word, tag = item[0],item[1]
            word_id, tag_id = w2i[word], t2i[tag]
            if i == 0:
                pi[tag_id] += 1
            if i != len(sequence) - 1:
                item_next = sequence[i + 1]
                tag_next = item_next[1]
                tag_next_id = t2i[tag_next]
                trans[tag_id][tag_next_id] += 1
            emit[tag_id][word_id] += 1
    
    pi /= np.sum(pi)  
    trans /= np.sum(trans, axis = 1).reshape(len(t2i.keys()), -1)
    emit /= np.sum(emit, axis = 1).reshape(len(t2i.keys()), -1)
    return pi, emit, trans


def main(train_in, word_to_index, tag_to_index, hmminit, hmmemit,hmmtrans):

    sequences = data_reading(train_in)
    w2i = index_reading(word_to_index)
    t2i = index_reading(tag_to_index)
    pi, emit, trans= hmm(sequences, w2i, t2i)
    np.savetxt(hmminit, pi)
    np.savetxt(hmmemit, emit)
    np.savetxt(hmmtrans, trans)

if __name__ == '__main__':
    '''
    mac用:
    python3 learnhmm.py ./en_data/train.txt ./en_data/index_to_word.txt \
        ./en_data/index_to_tag.txt ./en_data/hmmint.txt \
        ./en_data/hmmemit.txt ./en_data/hmmtrans.txt
    '''
    train_in = sys.argv[1]
    index_to_word = sys.argv[2]
    index_to_tag = sys.argv[3]
    hmminit = sys.argv[4]
    hmmemit = sys.argv[5]
    hmmtrans = sys.argv[6]

    main(train_in,index_to_word,index_to_tag, hmminit, hmmemit,hmmtrans)
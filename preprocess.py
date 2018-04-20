# -*- coding: utf-8 -*-
import os
import pickle
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help="data directory with a corpus.txt file (must have write access to this folder)", required=True)
    parser.add_argument('--window', type=int, default=5, help="window size")
    parser.add_argument('--max_word_len', type=int, default=24, help='ignore words longer than this')
    return parser.parse_args()

def to_str(lst):
    return ','.join([str(i) for i in lst])

class Preprocess(object):

    def __init__(self, window, data_dir):
        self.window = window
        #spl sym for words 
        self.unk = '<UNK>'
        self.bos = '<BOS>'
        self.eos = '<EOS>'
        #spl sym for chars
        self.bow = '<BOW>'
        self.eow = '<EOW>'
        self.pad = '<PAD>'
        self.spl_words = set([self.unk, self.bos, self.eos])
        self.data_dir = data_dir

    def skipgram(self, sentence, i):
        iword = sentence[i]
        left = sentence[max(i - self.window, 0): i]
        right = sentence[i + 1: i + 1 + self.window]
        bos_fill = [self.bos] * (self.window - len(left))
        #bos_fill = [self.unk] * (self.window - len(left))
        eos_fill = [self.eos] * (self.window - len(right))
        #eos_fill = [self.unk] * (self.window - len(right))
        return iword, bos_fill + left + right + eos_fill 

    def build(self, max_word_len=25):
        filepath = self.data_dir + '/corpus.txt'
        print("building vocab...", filepath)
        line_num = 0
        self.wc = {}
        total_words = 0
        with open(filepath, 'r', encoding= 'utf-8') as file:
            for line in file:
                line_num += 1
                if not line_num % 1000:
                    print("working on {}kth line".format(line_num // 1000))
                line = line.strip()
                if not line:
                    continue
                sent = [w for w in line.split() if len(w) + 2 < max_word_len]
                for word in sent:
                    self.wc[word] = self.wc.get(word, 0) + 1
                    total_words += 1
        print("")
        print("total word types", len(self.wc))
        self.vocab = [self.unk, self.bos, self.eos] + sorted(self.wc, key=self.wc.get, reverse=True) #all word types frequency sorted
        self.idx2word = {idx:word for idx,word in enumerate(self.vocab)}
        self.word2idx = {word:idx for idx, word in self.idx2word.items()}
        print("total word types with spl symbols", len(self.word2idx))
        self.idx2unigram_prob = {idx: float(self.wc.get(word, 0))/ float(total_words) for idx,word in self.idx2word.items()}


        self.wordidx2charidx = {} 
        self.char2idx = {self.pad : 0, self.bow: 1, self.eow: 2, self.unk: 4, self.bos: 5, self.eos: 6}
        for word,idx in self.word2idx.items():
            char_idx_lst = [self.char2idx[self.bow]]
            if word not in self.spl_words:
                for i in word:
                    if ord(i) < 127:
                        self.char2idx[i] = self.char2idx.get(i, len(self.char2idx))
                        char_idx_lst.append(self.char2idx[i])
                    else:
                        self.char2idx[self.unk] = self.char2idx.get(self.unk, len(self.char2idx))
                        char_idx_lst.append(self.char2idx[self.unk])
            else:
               self.char2idx[word] = self.char2idx.get(word, len(self.char2idx))
               char_idx_lst.append(self.char2idx[word])
            char_idx_lst.append(self.char2idx[self.eow])
            self.wordidx2charidx[idx] = char_idx_lst + \
                                        [self.char2idx[self.pad]] * (max_word_len - len(char_idx_lst)) +\
                                        [len(char_idx_lst)] #spelling with padding
            assert len(self.wordidx2charidx[idx]) == max_word_len + 1
        self.idx2char = {idx:c for c,idx in self.char2idx.items()}
        print("build done")
        print("saving files...")
        pickle.dump(self.vocab, open(os.path.join(self.data_dir, 'vocab.pkl'), 'wb'))
        pickle.dump(self.idx2unigram_prob, open(os.path.join(self.data_dir, 'idx2unigram_prob.pkl'), 'wb'))
        pickle.dump(self.idx2word, open(os.path.join(self.data_dir, 'idx2word.pkl'), 'wb'))
        pickle.dump(self.word2idx, open(os.path.join(self.data_dir, 'word2idx.pkl'), 'wb'))
        #pickle.dump(self.wordidx2f, open(os.path.join(self.data_dir, 'wordidx2f.pkl'), 'wb'))

        pickle.dump(self.char2idx, open(os.path.join(self.data_dir, 'char2idx.pkl'), 'wb'))
        pickle.dump(self.idx2char, open(os.path.join(self.data_dir, 'idx2char.pkl'), 'wb'))
        pickle.dump(self.wordidx2charidx, open(os.path.join(self.data_dir, 'wordidx2charidx.pkl'), 'wb'))

if __name__ == '__main__':
    args = parse_args()
    preprocess = Preprocess(window=args.window, data_dir=args.data_dir)
    preprocess.build(max_word_len = args.max_word_len)

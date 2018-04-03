# -*- coding: utf-8 -*-
import os
import pickle
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data/', help="data directory path")
    parser.add_argument('--vocab', type=str, default='./data/corpus.txt', help="corpus path for building vocab")
    parser.add_argument('--corpus', type=str, default='./data/corpus.txt', help="corpus path")
    parser.add_argument('--window', type=int, default=5, help="window size")
    parser.add_argument('--max_vocab', type=int, default=20000, help="maximum number of vocab")
    return parser.parse_args()

def to_str(lst):
    return ','.join([str(i) for i in lst])

class Preprocess(object):

    def __init__(self, window=5, data_dir='./data/'):
        self.window = window
        self.unk = '<UNK>'
        self.bos = '<BOS>'
        self.eos = '<EOS>'
        self.bow = '<BOW>'
        self.eow = '<EOW>'
        self.pad = '<PAD>'
        self.max_word_len = 0
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

    def add_pad(self):
        for w_idx, char_idx_lst in self.wordidx2charidx.items():
            p = [self.char2idx[self.pad]] * (self.max_word_len - len(char_idx_lst))
            self.wordidx2charidx[w_idx] = char_idx_lst + p
        return True

    def build(self, filepath, max_vocab=20000):
        print("building vocab...", filepath)
        line_num = 0
        self.wc = {self.bos: 1, self.eos: 1, self.unk: 1}
        with open(filepath, 'r', encoding= 'utf-8') as file:
            for line in file:
                line_num += 1
                if not line_num % 10000:
                    print("working on {}kth line".format(line_num // 10000))
                line = line.strip()
                if not line:
                    continue
                sent = line.split()
                for word in sent:
                    self.wc[word] = self.wc.get(word, 0) + 1
        print("")
        print("total word types", len(self.wc))
        print("max_vocab:", max_vocab)
        self.idx2word = [self.bos, self.eos, self.unk] + sorted(self.wc, key=self.wc.get, reverse=True)[:max_vocab - 1]
        #self.idx2word = [self.unk] + sorted(self.wc, key=self.wc.get, reverse=True)[:max_vocab - 1]
        self.word2idx = {self.idx2word[idx]: idx for idx, _ in enumerate(self.idx2word)}
        print("total word types after thresholding", len(self.idx2word))

        self.char2idx = {self.pad : 0, self.bow: 1, self.eow: 2, self.unk: 4, self.bos: 5, self.eos: 6}
        self.wordidx2charidx = {} 
        self.wordidx2len = {}
        for idx, word in enumerate(self.idx2word):
            if idx not in self.wordidx2charidx:
                char_idx_lst = [self.char2idx[self.bow]]
                if word not in self.char2idx:
                    for i in word:
                        self.char2idx[i] = self.char2idx.get(i, len(self.char2idx))
                        char_idx_lst.append(self.char2idx[i])
                else:
                    self.char2idx[word] = self.char2idx.get(word, len(self.char2idx))
                    char_idx_lst.append(self.char2idx[word])

                char_idx_lst.append(self.char2idx[self.eow])
                self.wordidx2charidx[idx] = char_idx_lst
                self.wordidx2len[idx] = len(char_idx_lst)
                self.max_word_len = self.max_word_len if len(char_idx_lst) < self.max_word_len else len(char_idx_lst)
            

        self.vocab = set([word for word in self.word2idx])
        pickle.dump(self.wc, open(os.path.join(self.data_dir, 'wc.pkl'), 'wb'))
        pickle.dump(self.vocab, open(os.path.join(self.data_dir, 'vocab.pkl'), 'wb'))
        pickle.dump(self.idx2word, open(os.path.join(self.data_dir, 'idx2word.pkl'), 'wb'))
        pickle.dump(self.word2idx, open(os.path.join(self.data_dir, 'word2idx.pkl'), 'wb'))
        pickle.dump(self.idx2word, open(os.path.join(self.data_dir, 'idx2word.pkl'), 'wb'))
        pickle.dump(self.char2idx, open(os.path.join(self.data_dir, 'char2idx.pkl'), 'wb'))

        self.idx2char = {idx:c for c,idx in self.char2idx.items()}
        pickle.dump(self.idx2char, open(os.path.join(self.data_dir, 'idx2char.pkl'), 'wb'))
        print("build done")
        print("padding word2char...")
        self.add_pad()
        pickle.dump(self.wordidx2charidx, open(os.path.join(self.data_dir, 'wordidx2charidx.pkl'), 'wb'))
        pickle.dump(self.wordidx2len, open(os.path.join(self.data_dir, 'wordidx2len.pkl'), 'wb'))

    def convert(self, filepath):
        print("converting corpus...")
        line_num = 0
        #train_instances = []
        train_instances = open(os.path.join(self.data_dir, 'train.txt'), 'wb')
        #train_char_instances = []
        #train_char_instances = open(os.path.join(self.data_dir, 'train_spelling.txt'), 'w')
        with open(filepath, 'r', encoding='utf-8') as file:
            for line in file:
                line_num += 1
                if not line_num % 10000:
                    print("working on {}kth line\r".format(line_num // 10000))
                line = line.strip()
                if not line:
                    continue
                sent = []
                for word in line.split():
                    if word in self.vocab:
                        sent.append(word)
                    else:
                        sent.append(self.unk)
                for i in range(len(sent)):
                    iword, owords = self.skipgram(sent, i)
                    iw_idx = self.word2idx[iword]
                    ow_idxs = [self.word2idx[oword] for oword in owords]
                    #train_instances.write(str(iw_idx) + '\t' + to_str(ow_idxs) + '\n') 
                    train_instances.write(bytes(to_str([iw_idx] + ow_idxs), 'utf-8'))
                    #ic_idxs = self.wordidx2charidx[iw_idx]
                    #ic_len = self.wordidx2len[iw_idx]
                    #oc_idxs = [self.wordidx2charidx[ow_idx] for ow_idx in ow_idxs]
                    #oc_lens = [self.wordidx2len[ow_idx] for ow_idx in ow_idxs]
                    #oc_idxs = list(oc_idxs)
                    #oc_lens = list(oc_lens)
                    #train_char_instances.write(to_str(ic_idxs) + '\t' + str(ic_len) + '\t' + to_str(oc_idxs) + '\t' + to_str( oc_lens) + '\n')
        print("")
        #pickle.dump(train_instances, open(os.path.join(self.data_dir, 'train.dat'), 'wb'))
        #pickle.dump(train_char_instances, open(os.path.join(self.data_dir, 'train_chars.dat'), 'wb'))
        train_instances.flush()
        train_instances.close()
        #train_char_instances.flush()
        #train_char_instances.close()
        print("conversion done")



if __name__ == '__main__':
    args = parse_args()
    preprocess = Preprocess(window=args.window, data_dir=args.data_dir)
    preprocess.build(args.vocab, max_vocab=args.max_vocab)
    preprocess.convert(args.corpus)

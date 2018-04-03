# -*- coding: utf-8 -*-

import torch
import os
import pickle
import random
import argparse
import torch as t
import numpy as np

from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from model import Word2Vec, SGNS, Spell2Vec, load_spelling
import linecache


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='sgns', help="model name")
    parser.add_argument('--data_dir', type=str, default='./data/', help="data directory path")
    parser.add_argument('--save_dir', type=str, default='./models/', help="model directory path")
    parser.add_argument('--e_dim', type=int, default=300, help="embedding dimension")
    parser.add_argument('--num_neg_samples', type=int, default=20, help="number of negative samples")
    parser.add_argument('--epoch', type=int, default=10, help="number of epochs")
    parser.add_argument('--batch_size', type=int, default=50, help="mini-batch size")
    parser.add_argument('--subsample_threshold', type=float, default=1e-5, help="subsample threshold")
    parser.add_argument('--resume', action='store_true', help="resume learning")
    parser.add_argument('--bidirectional', action='store_true', help="use bidirectional RNN for Spell2Vec")
    parser.add_argument('--use_noise_weights', action='store_true', help="use weights for negative sampling")
    parser.add_argument('--embedding_model', action='store',type=str, choices=set(['Word2Vec', 'Spell2Vec']), default='Word2Vec', help="which model to use")
    parser.add_argument('--window', action='store', type=int, default=5, help="context window size")
    parser.add_argument('--max_vocab', action='store', type=int, default=20000, help='max vocab size for word-level embeddings')
    parser.add_argument('--gpuid', type=int, default=-1, help="which gpu to use")
    return parser.parse_args()

def my_collate(batch):
    iwords, owords = zip(* batch)
    iwords = torch.LongTensor(np.concatenate(iwords))
    owords = torch.LongTensor(np.concatenate(owords)) 
    #target = torch.LongTensor(target)
    return [iwords, owords] #, target]

class LazyTextDataset(Dataset):
    def __init__(self, corpus_file, word2idx_file, window = 5, max_vocab=1e8):
        self.corpus_file = corpus_file 
        self.unk = '<UNK>'
        self.bos = '<BOS>'
        self.eos = '<EOS>'
        self.bow = '<BOW>'
        self.eow = '<EOW>'
        self.pad = '<PAD>'
        self.word2idx = pickle.load(open(word2idx_file, 'rb'))
        self.max_vocab = max_vocab if max_vocab < len(self.word2idx) else len(self.word2idx)
        self._total_data = 0
        self.window = window
        with open(self.corpus_file, "r", encoding="utf-8") as f:
            self._total_data = len(f.readlines()) - 1

    def skipgram_instances(self, sentence):
        instances = []
        sentence = [self.word2idx[word] \
                    if self.word2idx[word] < self.max_vocab else self.word2idx[self.unk] \
                    for word in sentence.strip().split() \
                    if word in self.word2idx]
        for i,iword in enumerate(sentence):
            left = sentence[max(i - self.window, 0): i]
            right = sentence[i + 1: i + 1 + self.window]
            bos_fill = [self.word2idx[self.bos]] * (self.window - len(left))
            eos_fill = [self.word2idx[self.eos]] * (self.window - len(right))
            context = bos_fill + left + right + eos_fill
            instances.append((iword, context))
        return instances

    def __getitem__(self, idx):
        line = linecache.getline(self.corpus_file, idx + 1)
        instances = self.skipgram_instances(line)
        iwords , owords = zip(*instances)
        iw, ows = np.array(list(iwords)), np.array(list(owords))
        return iw, ows

    def __len__(self):
        return self._total_data


def train(args):
    if args.gpuid > -1:
        torch.cuda.set_device(args.gpuid)
        tmp = torch.ByteTensor([0])
        tmp.cuda()
        print("using GPU", args.gpuid)
    else:
        print("using CPU")
    
    if args.use_noise_weights:
        idx2unigram_prob = pickle.load(open(os.path.join(args.data_dir, 'idx2unigram_prob.pkl'), 'rb'))
        idx, unigram_prob = zip(*sorted([(idx,p) for idx,p in idx2unigram_prob.items()]))
        max_noise_vocab = 50000 if args.max_vocab > 50000 else args.max_vocab
        unigram_prob = np.array(unigram_prob[:max_noise_vocab])
        noise_weights = np.clip(1 - np.sqrt(args.ss_t / unigram_prob), 0, 1)
    else:
        noise_weights = None

    if args.embedding_model == 'Word2Vec':
        embedding_model = Word2Vec(vocab_size=args.max_vocab, embedding_size=args.e_dim)
    elif args.embedding_model == 'Spell2Vec':
        char2idx = pickle.load(open(os.path.join(args.data_dir, 'char2idx.pkl'), 'rb'))
        wordidx2spelling, vocab_size = load_spelling(
                                        os.path.join(args.data_dir, 'wordidx2charidx.pkl'),
                                        os.path.join(args.data_dir, 'wordidx2len.pkl')
                                        )
        embedding_model = Spell2Vec(wordidx2spelling, 
                                    noise_vocab_size = len(noise_weights) if noise_weights is not None else 20000,
                                    char_vocab_size = len(char2idx), 
                                    embedding_size=args.e_dim,
                                    char_embedding_size=20,
                                    rnn_size=50,
                                    dropout=0.3,
                                    bidirectional=args.bidirectional)
        #dataset = PermutedSubsampledCharCorpus(os.path.join(args.data_dir, 'train_chars.pkl'))
        #dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    else:
        raise NotImplementedError('unknown embedding model')
    dataset = LazyTextDataset(corpus_file = os.path.join(args.data_dir, 'corpus.txt'), 
                              word2idx_file = os.path.join(args.data_dir, 'word2idx.pkl'), 
                              window = args.window,
                              max_vocab = args.max_vocab if args.embedding_model == 'Word2Vec' else 1e8)
    dataloader = DataLoader(dataset = dataset, 
                            batch_size = args.batch_size, 
                            shuffle = True, 
                            collate_fn = my_collate)
    total_batches = int(np.ceil(len(dataset) / args.batch_size))
    sgns = SGNS(embedding_model=embedding_model, num_neg_samples=args.num_neg_samples, weights= noise_weights)
    learnable_params = filter(lambda p: p.requires_grad, sgns.parameters()) 
    optim = Adam(learnable_params)
    if args.gpuid > -1:
        sgns = sgns.cuda()

    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)

    if args.resume:
        sgns.load_state_dict(t.load(os.path.join(args.save_dir, '{}.pt'.format(args.name))))
        optim.load_state_dict(t.load(os.path.join(args.save_dir, '{}.optim.pt'.format(args.name))))
    print(sgns)
    for epoch in range(1, args.epoch + 1):
        for batch_idx, batch in enumerate(dataloader):
            iword, owords = batch
            nwords = sgns.sample_noise(iword.size()[0])
            loss = sgns(iword, owords, nwords)
            optim.zero_grad()
            loss.backward()
            optim.step()
            print("[e{:2d}][b{:5d}/{:5d}] loss: {:7.4f}\r".format(epoch, batch_idx + 1, total_batches, loss.data[0]))
        print("")
        embedding_model.save_embeddings(args.save_dir)
        t.save(sgns.state_dict(), os.path.join(args.save_dir, '{}.pt'.format(args.name)))
        t.save(optim.state_dict(), os.path.join(args.save_dir, '{}.optim.pt'.format(args.name)))

if __name__ == '__main__':
    print(parse_args())
    train(parse_args())

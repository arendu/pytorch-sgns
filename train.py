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
from model import Word2Vec, SGNS, CharRNN2Vec, load_spelling
import pdb


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='sgns', help="model name")
    parser.add_argument('--data_dir', type=str, default='./data/', help="data directory path")
    parser.add_argument('--save_dir', type=str, default='./models/', help="model directory path")
    parser.add_argument('--e_dim', type=int, default=300, help="embedding dimension")
    parser.add_argument('--num_neg_samples', type=int, default=20, help="number of negative samples")
    parser.add_argument('--epoch', type=int, default=10, help="number of epochs")
    parser.add_argument('--batch_size', type=int, default=1024, help="mini-batch size")
    parser.add_argument('--ss_t', type=float, default=1e-5, help="subsample threshold")
    parser.add_argument('--resume', action='store_true', help="resume learning")
    parser.add_argument('--weights', action='store_true', help="use weights for negative sampling")
    parser.add_argument('--embedding_model', action='store',type=str, choices=set(['Word2Vec', 'CharRNN2Vec']), default='Word2Vec', help="which model to use")
    parser.add_argument('--gpuid', type=int, default=-1, help="which gpu to use")
    return parser.parse_args()


class PermutedSubsampledCharCorpus(Dataset):
    def __init__(self, datapath, ws=None):
        data = pickle.load(open(datapath, 'rb'))
        pdb.set_trace()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ichars, ilens, ochars, olens = self.data[idx]
        return np.array(ichars), ilens, np.array(ochars), np.array(olens)


class PermutedSubsampledCorpus(Dataset):
    def __init__(self, datapath, ws=None):
        data = pickle.load(open(datapath, 'rb'))
        if ws is not None:
            self.data = []
            for iword, owords, ichars, ochars in data:
                if random.random() > ws[iword]:
                    self.data.append((iword, owords))
        else:
            self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        iword, owords = self.data[idx]
        return iword, np.array(owords)


def train(args):
    if args.gpuid > -1:
        torch.cuda.set_device(args.gpuid)
        tmp = torch.ByteTensor([0])
        tmp.cuda()
        print("using GPU", args.gpuid)
    else:
        print("using CPU")

    if args.embedding_model == 'Word2Vec':
        idx2word = pickle.load(open(os.path.join(args.data_dir, 'idx2word.dat'), 'rb'))
        wc = pickle.load(open(os.path.join(args.data_dir, 'wc.dat'), 'rb'))
        wf = np.array([wc[word] for word in idx2word])
        wf = wf / wf.sum()
        ws = 1 - np.sqrt(args.ss_t / wf)
        ws = np.clip(ws, 0, 1)
        vocab_size = len(idx2word)
        weights = wf if args.weights else None
        embedding_model = Word2Vec(vocab_size=vocab_size, embedding_size=args.e_dim)
    elif args.embedding_model == 'CharRNN2Vec':
        char2idx = pickle.load(open(os.path.join(args.data_dir, 'char2idx.dat'), 'rb'))
        char_vocab_size = len(char2idx)
        wordidx2spelling, vocab_size = load_spelling(
                                        os.path.join(args.data_dir, 'wordidx2charidx.dat'),
                                        os.path.join(args.data_dir, 'wordidx2len.dat')
                                        )
        embedding_model = CharRNN2Vec(wordidx2spelling, 
                                        vocab_size,
                                        char_vocab_size, 
                                        embedding_size=args.e_dim)
        #dataset = PermutedSubsampledCharCorpus(os.path.join(args.data_dir, 'train_chars.dat'))
        #dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        weights = None
    else:
        raise NotImplementedError('unknown embedding model')
    dataset = PermutedSubsampledCorpus(os.path.join(args.data_dir, 'train.dat'))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    total_batches = int(np.ceil(len(dataset) / args.batch_size))
    sgns = SGNS(embedding_model=embedding_model, num_neg_samples=args.num_neg_samples, weights=weights)
    learnable_params = filter(lambda p: p.requires_grad, sgns.parameters()) 
    optim = Adam(learnable_params)
    num_pararams = sum([p.numel() for p in sgns.parameters() if p.requires_grad])
    #optim = Adam(sgns.parameters())
    if args.gpuid > -1:
        sgns = sgns.cuda()

    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)

    if args.resume:
        sgns.load_state_dict(t.load(os.path.join(args.save_dir, '{}.pt'.format(args.name))))
        optim.load_state_dict(t.load(os.path.join(args.save_dir, '{}.optim.pt'.format(args.name))))
    print(sgns)
    print('learnable_params', num_pararams)
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
    train(parse_args())

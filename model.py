# -*- coding: utf-8 -*-

import torch
import numpy as np
import torch.nn as nn
import os, pickle

from torch import LongTensor as LT
from torch import FloatTensor as FT
from torch.autograd import Variable as Var
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
import pdb

def t_sort(v):
    assert len(v.shape) == 1
    sorted_v, ind_v = torch.sort(v, 0, descending = True)
    return sorted_v, ind_v

def t_unsort(sort_idx):
    unsort_idx = torch.zeros_like(sort_idx).long().scatter_(0, sort_idx, torch.arange(sort_idx.size(0)).long())
    return unsort_idx


def t_tolist(v):
    if v.is_cuda:
        v_list = v.cpu().numpy().tolist()
    else:
        v_list = v.numpy().tolist()
    return v_list

def load_spelling(w2spath):
    wordidx2spelling = pickle.load(open(w2spath, 'rb'))
    vocab_size = len(wordidx2spelling)
    max_spelling_len = len(wordidx2spelling[0])
    spelling_emb = torch.nn.Embedding(vocab_size, max_spelling_len)
    t = torch.zeros(vocab_size, max_spelling_len)
    for w_idx, spelling in wordidx2spelling.items():
        t[w_idx] = torch.FloatTensor(spelling) 
    spelling_emb.weight = torch.nn.Parameter(t)
    spelling_emb.requires_grad = False
    return spelling_emb, len(wordidx2spelling)

def load_model(path):
    model = torch.load(path)
    return model

class Spell2Vec(nn.Module):
    def __init__(self, wordidx2spelling, 
                word_vocab_size,
                noise_vocab_size,
                char_vocab_size, 
                embedding_size=300, 
                char_embedding_size=20, 
                rnn_size=50, 
                padding_idx=0, 
                dropout=0.3, 
                bidirectional = False):
        super(Spell2Vec, self).__init__()
        self.char_vocab_size = char_vocab_size
        self.word_vocab_size = word_vocab_size
        self.noise_vocab_size = noise_vocab_size
        self.embedding_size = embedding_size
        self.char_embedding_size = char_embedding_size
        self.wordidx2spelling = wordidx2spelling
        self.dropout = nn.Dropout(dropout)
        self.input_char_embedding = nn.Embedding(self.char_vocab_size, char_embedding_size, padding_idx = padding_idx)
        self.context_char_embedding = nn.Embedding(self.char_vocab_size, char_embedding_size, padding_idx = padding_idx)
        self.input_char_embedding.weight = nn.Parameter(FT(self.char_vocab_size, char_embedding_size).uniform_(-0.5 / self.char_embedding_size, 0.5 / self.char_embedding_size))
        self.context_char_embedding.weight = nn.Parameter(FT(self.char_vocab_size, self.char_embedding_size).uniform_(-0.5 / self.char_embedding_size, 0.5 / self.char_embedding_size))
        self.input_rnn = nn.LSTM(char_embedding_size, rnn_size, dropout=dropout, bidirectional = bidirectional)
        self.context_rnn = nn.LSTM(char_embedding_size, rnn_size, dropout=dropout, bidirectional = bidirectional)
        self.input_linear = nn.Linear(rnn_size * (2 if bidirectional else 1), self.embedding_size)
        self.context_linear = nn.Linear(rnn_size * (2 if bidirectional else 1), self.embedding_size)
        #word-level embeddings
        self.ivectors = nn.Embedding(self.word_vocab_size, self.embedding_size, padding_idx=padding_idx)
        self.ovectors = nn.Embedding(self.word_vocab_size, self.embedding_size, padding_idx=padding_idx)
        #self.ivectors.weight = nn.Parameter(torch.cat([torch.zeros(1, self.embedding_size), FT(self.word_vocab_size - 1, self.embedding_size).uniform_(-0.5 / self.embedding_size, 0.5 / self.embedding_size)]))
        #self.ovectors.weight = nn.Parameter(torch.cat([torch.zeros(1, self.embedding_size), FT(self.word_vocab_size - 1, self.embedding_size).uniform_(-0.5 / self.embedding_size, 0.5 / self.embedding_size)]))
        self.ivectors.weight = nn.Parameter(FT(self.word_vocab_size, self.embedding_size).uniform_(-0.5 / self.embedding_size, 0.5 / self.embedding_size))
        self.ovectors.weight = nn.Parameter(FT(self.word_vocab_size, self.embedding_size).uniform_(-0.5 / self.embedding_size, 0.5 / self.embedding_size))
        self.ivectors.weight.requires_grad = True
        self.ovectors.weight.requires_grad = True


    def batch_rnn(self, data, lengths, char_rnn, char_embedding, linear):
        sorted_lengths, sorted_length_idx = t_sort(lengths)
        unsorted_length_idx = t_unsort(sorted_length_idx)
        sorted_data = data[sorted_length_idx]
        sorted_data = Var(sorted_data)
        sorted_embeddings = char_embedding(sorted_data)
        sorted_packed = pack(sorted_embeddings, t_tolist(sorted_lengths), batch_first=True)
        output, (ht,ct) = char_rnn(sorted_packed, None)
        #output = unpack(output)[0]
        del output, ct
        if ht.size(0) == 2:
            ht = torch.cat([ht[0, :, :], ht[1, :, :]], dim=1)  # concat the last ht from fwd RNN and first ht from bwd RNN
        else:
            ht = ht.squeeze()
        ht = linear(self.dropout(ht))
        ht_unsorted = ht[unsorted_length_idx]  # TODO:check if unsorting is working correctly
        del data, lengths, sorted_data, sorted_lengths, sorted_length_idx,sorted_embeddings,sorted_packed, ht
        return ht_unsorted

    def query(self, wordidx_list):
        self.eval()
        data = torch.LongTensor(wordidx_list)
        data = data.cuda() if self.wordidx2spelling.weight.is_cuda else data
        vecs = self.input_vectors(data)
        vecs = vecs.data.cpu().numpy()
        return vecs

    def input_vectors(self, data):
        data_idxs = torch.arange(data.size(0)).long()
        data_idxs = data_idxs.cuda() if data.is_cuda else data_idxs
        hf_data = data[data < self.word_vocab_size]
        hf_data_idxs = data_idxs[data < self.word_vocab_size]
        hf_data = Var(hf_data) #, requires_grad = False)
        hf_embeddings = self.ivectors(hf_data)
        if hf_data.size(0) < data.size(0):
            lf_data = data[data >= self.word_vocab_size]
            lf_data_idxs = data_idxs[data >= self.word_vocab_size]
            lf_data = Var(lf_data) #, requires_grad = False)
            spelling_data = self.wordidx2spelling(lf_data).data.clone().long()
            spelling = spelling_data[:,:-1]
            lengths = spelling_data[:,-1]
            lf_embeddings = self.batch_rnn(spelling, lengths, self.input_rnn, self.input_char_embedding, self.input_linear)
            embeddings = torch.cat([hf_embeddings, lf_embeddings], dim=0)
            f_idx= torch.cat([hf_data_idxs, lf_data_idxs], dim=0)
            embeddings[data_idxs,:] = embeddings[f_idx,:]
            #print('i', hf_data.size(0), lf_data.size(0))
            del spelling_data, spelling, lengths, f_idx, lf_data, lf_data_idxs, lf_embeddings
        else:
            embeddings = hf_embeddings
        del hf_data, data, data_idxs, hf_data_idxs, hf_embeddings
        return embeddings

    def context_vectors(self, data):
        bs,cs = data.shape
        data = data.contiguous()
        data = data.view(bs * cs)
        data_idxs = torch.arange(data.size(0)).long()
        data_idxs = data_idxs.cuda() if data.is_cuda else data_idxs
        hf_data = data[data < self.word_vocab_size]
        hf_data_idxs = data_idxs[data < self.word_vocab_size]
        hf_data = Var(hf_data) # , requires_grad = False)
        hf_embeddings = self.ovectors(hf_data)
        if hf_data.size(0) < data.size(0):
            lf_data = data[data >= self.word_vocab_size]
            lf_data_idxs = data_idxs[data >= self.word_vocab_size]
            lf_data = Var(lf_data) #, requires_grad = False)
            spelling_data = self.wordidx2spelling(lf_data).data.clone().long()
            spelling = spelling_data[:,:-1]
            lengths = spelling_data[:,-1]
            lf_embeddings = self.batch_rnn(spelling, lengths, self.context_rnn, self.context_char_embedding, self.context_linear)
            embeddings = torch.cat([hf_embeddings, lf_embeddings], dim=0)
            f_idx= torch.cat([hf_data_idxs, lf_data_idxs], dim=0)
            embeddings[data_idxs,:] = embeddings[f_idx,:]
            #print('c', hf_data.size(0), lf_data.size(0))
            del spelling_data, spelling, lengths, f_idx, lf_data, lf_data_idxs, lf_embeddings
        else:
            embeddings = hf_embeddings
        embeddings = embeddings.view(bs, cs, self.embedding_size)
        del hf_data, data, data_idxs, hf_data_idxs, hf_embeddings
        return embeddings

    def forward(self, data, is_input = True):
        #data = Var(data, requires_grad=False) # we make it a Var just so that it can be used with an embedding layer
        data = data.cuda() if self.wordidx2spelling.weight.is_cuda else data
        if is_input:
            #data = {batch_size, max_seq_len}
            #data_lengths = {batch_size}
            return self.input_vectors(data)
        else:
            #data = {batch_size, 2 x window_size, max_seq_len}
            #data_lengths = {batch_size, 2 x window_size}
            return self.context_vectors(data)
    
    def save_model(self, path):
        torch.save(self, path)

class Word2Vec(nn.Module):
    def __init__(self, word_vocab_size=20000, embedding_size=300, padding_idx=0):
        super(Word2Vec, self).__init__()
        self.word_vocab_size = word_vocab_size
        self.noise_vocab_size = word_vocab_size
        self.embedding_size = embedding_size
        self.ivectors = nn.Embedding(self.word_vocab_size, self.embedding_size, padding_idx=padding_idx)
        self.ovectors = nn.Embedding(self.word_vocab_size, self.embedding_size, padding_idx=padding_idx)
        #self.ivectors.weight = nn.Parameter(torch.cat([torch.zeros(1, self.embedding_size), FT(self.word_vocab_size - 1, self.embedding_size).uniform_(-0.5 / self.embedding_size, 0.5 / self.embedding_size)]))
        #self.ovectors.weight = nn.Parameter(torch.cat([torch.zeros(1, self.embedding_size), FT(self.word_vocab_size - 1, self.embedding_size).uniform_(-0.5 / self.embedding_size, 0.5 / self.embedding_size)]))
        self.ivectors.weight = nn.Parameter(FT(self.word_vocab_size, self.embedding_size).uniform_(-0.5 / self.embedding_size, 0.5 / self.embedding_size))
        self.ovectors.weight = nn.Parameter(FT(self.word_vocab_size, self.embedding_size).uniform_(-0.5 / self.embedding_size, 0.5 / self.embedding_size))
        self.ivectors.weight.requires_grad = True
        self.ovectors.weight.requires_grad = True

    def forward(self, data, lengths = None, is_input = True):
        if is_input:
            return self.input_vectors(data)
        else:
            return self.context_vectors(data)
    
    def query(self, wordidx_list):
        self.eval()
        vecs = self.input_vectors(wordidx_list)
        vecs = vecs.data.cpu().numpy()
        return vecs


    def input_vectors(self, data):
        #data = {batch_size}
        v = Var(LT(data)) #, requires_grad=False)
        v = v.cuda() if self.ivectors.weight.is_cuda else v
        vecs = self.ivectors(v)
        return vecs

    def context_vectors(self, data):
        #data = {batch_size x 2 * window_size}
        v = Var(LT(data)) #, requires_grad=False)
        v = v.cuda() if self.ivectors.weight.is_cuda else v
        vecs = self.ovectors(v)
        return vecs

    def save_model(self, path):
        torch.save(self, path)


class SGNS(nn.Module):
    def __init__(self, embedding_model, num_neg_samples=20, window_size =5, weights=None):
        super(SGNS, self).__init__()
        self.embedding_model = embedding_model
        self.num_neg_samples = num_neg_samples
        self.context_size = window_size * 2
        self.weights = None
        if weights is not None:
            wf = np.power(weights, 0.75)
            wf = wf / wf.sum()
            self.weights = FT(wf)
    
    def sample_noise(self, batch_size):
        if self.weights is not None:
            nwords = torch.multinomial(self.weights, batch_size * self.context_size * self.num_neg_samples, replacement=True).view(batch_size, -1)
        else:
            nwords = FT(batch_size, self.context_size * self.num_neg_samples).uniform_(0, self.embedding_model.noise_vocab_size - 1).long()
        return nwords


    def forward(self, iword, owords, nwords):
        ivectors = self.embedding_model(iword, is_input = True).unsqueeze(2)
        ovectors = self.embedding_model(owords, is_input = False)
        nvectors = -self.embedding_model(nwords, is_input = False)
        nll = -nn.functional.logsigmoid(torch.bmm(ovectors, ivectors).squeeze()).mean(1) #log_prob of belonging in "true" class
        nll_negated_noise = -nn.functional.logsigmoid(torch.bmm(nvectors, ivectors).squeeze()).view(-1, self.context_size, self.num_neg_samples).sum(2).mean(1) #log_prob of "noise" class
        loss = (nll + nll_negated_noise).mean()
        return loss 

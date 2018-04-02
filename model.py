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

def t_sort(v):
    assert len(v.shape) == 1
    sorted_v, ind_v = torch.sort(v, 0, descending = True)
    return sorted_v, ind_v

def t_unsort(sorted_v, ind_v):
    assert len(sorted_v.shape) == 1
    v = torch.zeros_like(sorted_v)
    v.scatter_(0, ind_v, sorted_v)
    return v

def t_tolist(v):
    if v.is_cuda:
        v_list = v.cpu().numpy().tolist()
    else:
        v_list = v.numpy().tolist()
    return v_list

def load_spelling(w2spath, w2lpath):
    wordidx2spelling = pickle.load(open(w2spath, 'rb'))
    wordidx2len = pickle.load(open(w2lpath, 'rb'))
    vocab_size = len(wordidx2spelling)
    max_spelling_len = len(wordidx2spelling[0])
    spelling_emb = torch.nn.Embedding(vocab_size, max_spelling_len + 1)
    t = torch.zeros(vocab_size, max_spelling_len + 1)
    for w_idx, spelling in wordidx2spelling.items():
        w_len = wordidx2len[w_idx]
        t[w_idx] = torch.FloatTensor(spelling + [w_len]) 
    spelling_emb.weight = torch.nn.Parameter(t)
    spelling_emb.requires_grad = False
    return spelling_emb, len(wordidx2spelling)


class Spell2Vec(nn.Module):
    def __init__(self, wordidx2spelling, vocab_size, char_vocab_size, embedding_size=100, char_embedding_size=20,padding_idx=0, dropout=0.3):
        super(Spell2Vec, self).__init__()
        self.char_vocab_size = char_vocab_size
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.char_embedding_size = char_embedding_size
        self.wordidx2spelling = wordidx2spelling
        self.input_char_embedding = nn.Embedding(self.char_vocab_size, char_embedding_size, padding_idx = padding_idx)
        self.context_char_embedding = nn.Embedding(self.char_vocab_size, char_embedding_size, padding_idx = padding_idx)
        self.input_rnn = nn.GRU(char_embedding_size, embedding_size, dropout=0.3)
        self.context_rnn = nn.GRU(char_embedding_size, embedding_size, dropout=0.3)

    def batch_rnn(self, data, lengths, char_rnn, char_embedding):
        sorted_lengths, sorted_length_idx = t_sort(lengths) 
        sorted_data = data[sorted_length_idx]
        sorted_data = Var(sorted_data)
        sorted_embeddings = char_embedding(sorted_data)
        sorted_packed = pack(sorted_embeddings, t_tolist(sorted_lengths), batch_first=True)
        _, ht = char_rnn(sorted_packed, None)
        ht  = ht.squeeze()
        ht_unsorted = ht[sorted_length_idx]
        del data,lengths,sorted_data, sorted_lengths, sorted_length_idx,sorted_embeddings,sorted_packed, ht, _
        return ht_unsorted

    def forward(self, data, is_input = True):
        data = Var(data, requires_grad=False) # we make it a Var just so that it can be used with an embedding layer
        data = data.cuda() if self.wordidx2spelling.weight.is_cuda else data
        data = self.wordidx2spelling(data).data.clone().long()
        if is_input:
            #data = {batch_size, max_seq_len}
            #data_lengths = {batch_size}
            spelling = data[:,:-1]
            lengths = data[:,-1]
            ht_unsorted = self.batch_rnn(spelling, lengths, self.input_rnn, self.input_char_embedding)
            return ht_unsorted
        else:
            #data = {batch_size, 2 x window_size, max_seq_len}
            #data_lengths = {batch_size, 2 x window_size}
            spelling = data[:,:,:-1]
            lengths = data[:,:,-1]
            spelling = spelling.contiguous()
            lengths = lengths.contiguous()
            bs, cs, max_seq_len = spelling.shape
            spelling = spelling.view(bs * cs, max_seq_len)
            lengths = lengths.view(bs * cs)
            ht_unsorted = self.batch_rnn(spelling, lengths, self.context_rnn, self.context_char_embedding)
            #ht_unsorted = {batch_size * 2 * window_size, embed_size}
            ht_unsorted = ht_unsorted.view(bs, cs, self.embedding_size)
            return ht_unsorted
    
    def save_embeddings(self, path):
        pass
            

class Word2Vec(nn.Module):
    def __init__(self, vocab_size=20000, embedding_size=300, padding_idx=0):
        super(Word2Vec, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.ivectors = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=padding_idx)
        self.ovectors = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=padding_idx)
        self.ivectors.weight = nn.Parameter(torch.cat([torch.zeros(1, self.embedding_size), FT(self.vocab_size - 1, self.embedding_size).uniform_(-0.5 / self.embedding_size, 0.5 / self.embedding_size)]))
        self.ovectors.weight = nn.Parameter(torch.cat([torch.zeros(1, self.embedding_size), FT(self.vocab_size - 1, self.embedding_size).uniform_(-0.5 / self.embedding_size, 0.5 / self.embedding_size)]))
        self.ivectors.weight.requires_grad = True
        self.ovectors.weight.requires_grad = True

    def forward(self, data, lengths = None, is_input = True):
        if is_input:
            return self.input_vectors(data)
        else:
            return self.context_vectors(data)

    def input_vectors(self, data):
        #data = {batch_size}
        v = Var(LT(data), requires_grad=False)
        v = v.cuda() if self.ivectors.weight.is_cuda else v
        vecs = self.ivectors(v)
        return vecs

    def context_vectors(self, data):
        #data = {batch_size x 2 * window_size}
        v = Var(LT(data), requires_grad=False)
        v = v.cuda() if self.ivectors.weight.is_cuda else v
        vecs = self.ovectors(v)
        return vecs

    def save_embeddings(self, path):
        idx2ivec = self.ivectors.weight.data.cpu().numpy()
        pickle.dump(idx2ivec, open(os.path.join(path, 'idx2ivec.dat'), 'wb'))
        idx2ovec = self.ovectors.weight.data.cpu().numpy()
        pickle.dump(idx2ovec, open(os.path.join(path, 'idx2ovec.dat'), 'wb'))


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
            nwords = FT(batch_size, self.context_size * self.num_neg_samples).uniform_(0, self.embedding_model.vocab_size - 1).long()
        return nwords


    def forward(self, iword, owords, nwords):
        ivectors = self.embedding_model(iword, is_input = True).unsqueeze(2)
        ovectors = self.embedding_model(owords, is_input = False)
        nvectors = -self.embedding_model(nwords, is_input = False)
        nll = -nn.functional.logsigmoid(torch.bmm(ovectors, ivectors).squeeze()).mean(1) #log_prob of belonging in "true" class
        nll_negated_noise = -nn.functional.logsigmoid(torch.bmm(nvectors, ivectors).squeeze()).view(-1, self.context_size, self.num_neg_samples).sum(2).mean(1) #log_prob of "noise" class
        loss = (nll + nll_negated_noise).mean()
        return loss 

    def __forward_original(self, iword, owords):
        batch_size = iword.size()[0]
        context_size = owords.size()[1]
        if self.weights is not None:
            nwords = torch.multinomial(self.weights, batch_size * context_size * self.num_neg_samples, replacement=True).view(batch_size, -1)
        else:
            nwords = FT(batch_size, context_size * self.num_neg_samples).uniform_(0, self.vocab_size - 1).long()
        ivectors = self.word_embedding_model(iword, is_input = True).unsqueeze(2)
        ovectors = self.word_embedding_model(owords, is_input = False)
        nvectors = self.word_embedding_model(nwords, is_input = False) * -1
        oloss = torch.bmm(ovectors, ivectors).squeeze().sigmoid().log().mean(1) #log_prob of belonging in "true" class
        nloss = torch.bmm(nvectors, ivectors).squeeze().sigmoid().log().view(-1, context_size, self.num_neg_samples).sum(2).mean(1) #log_prob of "noise" class
        del iword, owords, nwords, ivectors, ovectors, nvectors
        return -(oloss + nloss).mean()

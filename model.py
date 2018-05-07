# -*- coding: utf-8 -*-

import torch
import numpy as np
import torch.nn as nn
import os
import pickle


from torch import LongTensor as LT
from torch import FloatTensor as FT
from torch.autograd import Variable as Var
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack


def t_sort(v):
    assert len(v.shape) == 1
    sorted_v, ind_v = torch.sort(v, 0, descending=True)
    return sorted_v, ind_v


def t_unsort(sort_idx):
    unsort_idx = torch.zeros_like(sort_idx).type_as(sort_idx).long()
    unsort_idx = unsort_idx.scatter_(0, sort_idx, torch.arange(sort_idx.size(0)).type_as(sort_idx).long())
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
    return spelling_emb, len(wordidx2spelling), max_spelling_len


def load_model(path):
    model = torch.load(path)
    return model


class SpellHybrid2Vec(nn.Module):
    def __init__(self, wordidx2spelling,
                 word_vocab_size,
                 noise_vocab_size,
                 char_vocab_size,
                 embedding_size=300,
                 char_embedding_size=20,
                 padding_idx=0,
                 dropout=0.3,
                 bidirectional=False,
                 char_composition='RNN'):
        super(SpellHybrid2Vec, self).__init__()
        self.char_vocab_size = char_vocab_size
        self.word_vocab_size = word_vocab_size
        self.noise_vocab_size = noise_vocab_size
        self.embedding_size = embedding_size
        self.char_embedding_size = char_embedding_size
        self.wordidx2spelling = wordidx2spelling
        self.dropout = nn.Dropout(dropout)
        self.char_embedding = nn.Embedding(self.char_vocab_size, char_embedding_size, padding_idx=padding_idx)
        self.char_composition = char_composition
        self.char_embedding.weight = nn.Parameter(FT(
            self.char_vocab_size, char_embedding_size).uniform_(
                -0.5 / self.char_embedding_size, 0.5 / self.char_embedding_size))
        if self.char_composition == 'RNN':
            assert self.embedding_size % 2 == 0
            self.rnn_size = self.embedding_size // (2 if bidirectional else 1)
            self.char_rnn = nn.LSTM(char_embedding_size, self.rnn_size, dropout=dropout, bidirectional=bidirectional)
            # self.linear = nn.Linear(rnn_size * (2 if bidirectional else 1), self.embedding_size)
        elif self.char_composition == 'CNN':
            assert self.embedding_size % 4 == 0
            self.c1d_3g = torch.nn.Conv1d(self.char_embedding_size, self.embedding_size // 4, 3)
            self.c1d_4g = torch.nn.Conv1d(self.char_embedding_size, self.embedding_size // 4, 4)
            self.c1d_5g = torch.nn.Conv1d(self.char_embedding_size, self.embedding_size // 4, 5)
            self.c1d_6g = torch.nn.Conv1d(self.char_embedding_size, self.embedding_size // 4, 6)
        else:
            raise BaseException("unknown char_composition")
        # word-level embeddings
        self.ivectors = nn.Embedding(self.word_vocab_size, self.embedding_size, padding_idx=padding_idx)
        self.ovectors = nn.Embedding(self.word_vocab_size, self.embedding_size, padding_idx=padding_idx)
        self.ivectors.weight = nn.Parameter(FT(
            self.word_vocab_size, self.embedding_size).uniform_(-0.5 / self.embedding_size, 0.5 / self.embedding_size))
        self.ovectors.weight = nn.Parameter(FT(
            self.word_vocab_size, self.embedding_size).uniform_(-0.5 / self.embedding_size, 0.5 / self.embedding_size))
        self.ivectors.weight.requires_grad = True
        self.ovectors.weight.requires_grad = True
        freq_gate = nn.Sequential(nn.Linear(1, 1), nn.Sigmoid())

    def init_cuda(self,):
        print('init_cuda in SpellHybrid2Vec')

    def batch_cnn(self, data):
        data = Var(data)
        embeddings = self.dropout(self.char_embedding(data))
        embeddings = embeddings.transpose(1, 2)
        m_3g = torch.max(self.c1d_3g(embeddings), dim=2)[0]
        m_4g = torch.max(self.c1d_4g(embeddings), dim=2)[0]
        m_5g = torch.max(self.c1d_5g(embeddings), dim=2)[0]
        m_6g = torch.max(self.c1d_6g(embeddings), dim=2)[0]
        word_embeddings = torch.cat([m_3g, m_4g, m_5g, m_6g], dim=1)
        del m_3g, m_4g, m_5g, m_6g
        return word_embeddings

    def batch_rnn(self, data, lengths):
        sorted_lengths, sorted_length_idx = torch.sort(lengths, 0, True)
        unsorted_length_idx = t_unsort(sorted_length_idx)
        sorted_data = data[sorted_length_idx]
        sorted_data = Var(sorted_data)
        sorted_embeddings = self.dropout(self.char_embedding(sorted_data))

        sorted_packed = pack(sorted_embeddings, t_tolist(sorted_lengths), batch_first=True)
        output, (ht, ct) = self.char_rnn(sorted_packed, None)
        # output = unpack(output)[0]
        del output, ct
        if ht.size(0) == 2:
            # concat the last ht from fwd RNN and first ht from bwd RNN
            ht = torch.cat([ht[0, :, :], ht[1, :, :]], dim=1)
        else:
            ht = ht.squeeze()
        # ht = self.linear(self.dropout(ht))
        ht_unsorted = ht[unsorted_length_idx]  # TODO:check if unsorting is working correctly
        del data, lengths, sorted_data, sorted_lengths, sorted_length_idx, sorted_embeddings, sorted_packed, ht
        return ht_unsorted

    def query(self, word_idx, spelling):
        assert isinstance(word_idx, int)
        assert isinstance(spelling, list)
        self.eval()
        word_idx = torch.LongTensor([word_idx])
        length = torch.LongTensor([len(spelling)])
        spelling = torch.LongTensor(spelling).unsqueeze(0)
        if self.ivectors.weight.is_cuda:
            word_idx = word_idx.cuda()
            spelling = spelling.cuda()
            length = length.cuda()
        word_embedding = self.ivectors(Var(word_idx))
        if self.char_composition == 'RNN':
            composed_embedding = self.batch_rnn(spelling, length)
        elif self.char_composition == 'CNN':
            composed_embedding = self.batch_cnn(spelling)
        else:
            raise BaseException("unknown char_composition")
        embedding = word_embedding + composed_embedding
        vecs = embedding.data.cpu().numpy()
        return vecs

    def input_vectors(self, data):
        data_idxs = torch.arange(data.size(0)).long()
        data_idxs = data_idxs.cuda() if data.is_cuda else data_idxs
        hf_data = data
        hf_data = Var(hf_data.clone())
        hf_data[hf_data >= self.word_vocab_size] = 0
        hf_embeddings = self.ivectors(hf_data)
        lf_data = data
        lf_data = Var(lf_data)
        spelling_data = self.wordidx2spelling(lf_data).data.clone().long()
        spelling = spelling_data[:, :-1]
        lengths = spelling_data[:, -1]
        if self.char_composition == 'RNN':
            lf_embeddings = self.batch_rnn(spelling, lengths)
        elif self.char_composition == 'CNN':
            lf_embeddings = self.batch_cnn(spelling)
        else:
            raise BaseException("unknown char_composition")
        # embeddings = torch.cat([hf_embeddings, lf_embeddings], dim=0)
        # f_idx= torch.cat([hf_data_idxs, lf_data_idxs], dim=0)
        # embeddings[data_idxs,:] = embeddings[f_idx,:]
        # print('i', hf_data.size(0), lf_data.size(0))
        embeddings = hf_embeddings + lf_embeddings
        del spelling_data, spelling, lengths, lf_data, lf_embeddings
        del hf_data, data, data_idxs, hf_embeddings
        return embeddings

    def context_vectors(self, data):
        # data = {batch_size x 2 * window_size}
        bs, cs = data.shape
        data = data.view(bs * cs)
        hf_data = Var(data)  # , requires_grad=False)
        hf_data[hf_data >= self.word_vocab_size] = 0
        embeddings = self.ovectors(hf_data)
        embeddings = embeddings.view(bs, cs, self.embedding_size)
        del hf_data, data
        return embeddings

    def forward(self, data, is_input=True):
        # data = Var(data, requires_grad=False)  # we make it a Var just so that it can be used with an embedding layer
        data = data.cuda() if self.wordidx2spelling.weight.is_cuda else data
        if is_input:
            # data = {batch_size, max_seq_len}
            # data_lengths = {batch_size}
            return self.input_vectors(data)
        else:
            # data = {batch_size, 2 x window_size, max_seq_len}
            # data_lengths = {batch_size, 2 x window_size}
            return self.context_vectors(data)

    def save_model(self, path):
        torch.save(self, path)


class Spell2Vec(nn.Module):
    def __init__(self, wordidx2spelling,
                 word_vocab_size,
                 noise_vocab_size,
                 char_vocab_size,
                 embedding_size=300,
                 char_embedding_size=20,
                 padding_idx=0,
                 dropout=0.3,
                 char_composition='RNN',
                 bidirectional=False):
        super(Spell2Vec, self).__init__()
        self.char_vocab_size = char_vocab_size
        self.word_vocab_size = word_vocab_size
        self.noise_vocab_size = noise_vocab_size
        self.embedding_size = embedding_size
        self.char_embedding_size = char_embedding_size
        self.wordidx2spelling = wordidx2spelling
        self.dropout = nn.Dropout(dropout)
        self.char_composition = char_composition
        self.input_char_embedding = nn.Embedding(self.char_vocab_size, char_embedding_size, padding_idx=padding_idx)
        self.context_char_embedding = nn.Embedding(self.char_vocab_size, char_embedding_size, padding_idx=padding_idx)
        self.input_char_embedding.weight = nn.Parameter(FT(
            self.char_vocab_size, char_embedding_size).uniform_(
                -0.5 / self.char_embedding_size, 0.5 / self.char_embedding_size))
        self.context_char_embedding.weight = nn.Parameter(FT(
            self.char_vocab_size, self.char_embedding_size).uniform_(
                -0.5 / self.char_embedding_size, 0.5 / self.char_embedding_size))
        if self.char_composition == 'RNN':
            assert self.embedding_size % 2 == 0
            self.rnn_size = self.embedding_size // (2 if bidirectional else 1)
            self.input_rnn = nn.LSTM(char_embedding_size, self.rnn_size, dropout=dropout, bidirectional=bidirectional)
            self.context_rnn = nn.LSTM(char_embedding_size, self.rnn_size, dropout=dropout, bidirectional=bidirectional)
            # self.input_linear = nn.Linear(rnn_size * (2 if bidirectional else 1), self.embedding_size)
            # self.context_linear = nn.Linear(rnn_size * (2 if bidirectional else 1), self.embedding_size)
        elif self.char_composition == 'CNN':
            assert self.embedding_size % 4 == 0
            self.input_c1d_3g = torch.nn.Conv1d(self.char_embedding_size, self.embedding_size // 4, 3)
            self.input_c1d_4g = torch.nn.Conv1d(self.char_embedding_size, self.embedding_size // 4, 4)
            self.input_c1d_5g = torch.nn.Conv1d(self.char_embedding_size, self.embedding_size // 4, 5)
            self.input_c1d_6g = torch.nn.Conv1d(self.char_embedding_size, self.embedding_size // 4, 6)
            self.context_c1d_3g = torch.nn.Conv1d(self.char_embedding_size, self.embedding_size // 4, 3)
            self.context_c1d_4g = torch.nn.Conv1d(self.char_embedding_size, self.embedding_size // 4, 4)
            self.context_c1d_5g = torch.nn.Conv1d(self.char_embedding_size, self.embedding_size // 4, 5)
            self.context_c1d_6g = torch.nn.Conv1d(self.char_embedding_size, self.embedding_size // 4, 6)
        else:
            raise BaseException("unknown char_composition")
        # word-level embeddings
        self.ivectors = nn.Embedding(self.word_vocab_size, self.embedding_size, padding_idx=padding_idx)
        self.ovectors = nn.Embedding(self.word_vocab_size, self.embedding_size, padding_idx=padding_idx)
        self.ivectors.weight = nn.Parameter(FT(
            self.word_vocab_size, self.embedding_size).uniform_(-0.5 / self.embedding_size, 0.5 / self.embedding_size))
        self.ovectors.weight = nn.Parameter(FT(
            self.word_vocab_size, self.embedding_size).uniform_(-0.5 / self.embedding_size, 0.5 / self.embedding_size))
        self.ivectors.weight.requires_grad = True
        self.ovectors.weight.requires_grad = True

    def init_cuda(self,):
        pass

    def batch_cnn(self, data, char_embedding, c1d_3g, c1d_4g, c1d_5g, c1d_6g):
        data = Var(data)
        embeddings = self.dropout(char_embedding(data))
        embeddings = embeddings.transpose(1, 2)
        m_3g = torch.max(c1d_3g(embeddings), dim=2)[0]
        m_4g = torch.max(c1d_4g(embeddings), dim=2)[0]
        m_5g = torch.max(c1d_5g(embeddings), dim=2)[0]
        m_6g = torch.max(c1d_6g(embeddings), dim=2)[0]
        word_embeddings = torch.cat([m_3g, m_4g, m_5g, m_6g], dim=1)
        del m_3g, m_4g, m_5g, m_6g
        return word_embeddings

    def batch_rnn(self, data, lengths, char_rnn, char_embedding):
        sorted_lengths, sorted_length_idx = t_sort(lengths)
        unsorted_length_idx = t_unsort(sorted_length_idx)
        sorted_data = data[sorted_length_idx]
        sorted_data = Var(sorted_data)
        sorted_embeddings = char_embedding(sorted_data)
        sorted_packed = pack(sorted_embeddings, t_tolist(sorted_lengths), batch_first=True)
        output, (ht, ct) = char_rnn(sorted_packed, None)
        # output = unpack(output)[0]
        del output, ct
        if ht.size(0) == 2:
            # concat the last ht from fwd RNN and first ht from bwd RNN
            ht = torch.cat([ht[0, :, :], ht[1, :, :]], dim=1)
        else:
            ht = ht.squeeze()
        # ht = linear(self.dropout(ht))
        ht_unsorted = ht[unsorted_length_idx]  # TODO:check if unsorting is working correctly
        del data, lengths, sorted_data, sorted_lengths, sorted_length_idx, sorted_embeddings, sorted_packed, ht
        return ht_unsorted

    def query(self, word_idx, spelling):
        assert isinstance(word_idx, int)
        assert isinstance(spelling, list)
        self.eval()
        word_idx = torch.LongTensor([word_idx])
        length = torch.LongTensor([len(spelling)])
        spelling = torch.LongTensor(spelling).unsqueeze(0)
        if self.ivectors.weight.is_cuda:
            word_idx = word_idx.cuda()
            spelling = spelling.cuda()
        if 0 < word_idx[0] < self.word_vocab_size:
            embedding = self.ivectors(Var(word_idx))
        else:
            if self.char_composition == 'RNN':
                embedding = self.batch_rnn(spelling, length,
                                           self.input_rnn,
                                           self.input_char_embedding
                                           )
            elif self.char_composition == 'CNN':
                embedding = self.batch_cnn(spelling, self.input_char_embedding,
                                           self.input_c1d_3g,
                                           self.input_c1d_4g,
                                           self.input_c1d_5g,
                                           self.input_c1d_6g)
            else:
                raise BaseException("unknown char_composition")
        vecs = embedding.data.cpu().numpy()
        return vecs

    def input_vectors(self, data):
        data_idxs = torch.arange(data.size(0)).long()
        data_idxs = data_idxs.cuda() if data.is_cuda else data_idxs
        hf_data = data[data < self.word_vocab_size]
        hf_data_idxs = data_idxs[data < self.word_vocab_size]
        hf_data = Var(hf_data)  # , requires_grad = False)
        hf_embeddings = self.ivectors(hf_data)
        if hf_data.size(0) < data.size(0):
            lf_data = data[data >= self.word_vocab_size]
            lf_data_idxs = data_idxs[data >= self.word_vocab_size]
            lf_data = Var(lf_data)  # , requires_grad = False)
            spelling_data = self.wordidx2spelling(lf_data).data.clone().long()
            spelling = spelling_data[:, :-1]
            lengths = spelling_data[:, -1]
            if self.char_composition == 'RNN':
                lf_embeddings = self.batch_rnn(spelling, lengths,
                                               self.input_rnn,
                                               self.input_char_embedding
                                               )
            elif self.char_composition == 'CNN':
                lf_embeddings = self.batch_cnn(spelling, self.input_char_embedding,
                                               self.input_c1d_3g,
                                               self.input_c1d_4g,
                                               self.input_c1d_5g,
                                               self.input_c1d_6g)
            else:
                raise BaseException("unknown char_composition")
            embeddings = torch.cat([hf_embeddings, lf_embeddings], dim=0)
            f_idx = torch.cat([hf_data_idxs, lf_data_idxs], dim=0)
            embeddings[data_idxs, :] = embeddings[f_idx, :]
            # print('i', hf_data.size(0), lf_data.size(0))
            del spelling_data, spelling, lengths, f_idx, lf_data, lf_data_idxs, lf_embeddings
        else:
            embeddings = hf_embeddings
        del hf_data, data, data_idxs, hf_data_idxs, hf_embeddings
        return embeddings

    def context_vectors(self, data):
        bs, cs = data.shape
        data = data.contiguous()
        data = data.view(bs * cs)
        data_idxs = torch.arange(data.size(0)).long()
        data_idxs = data_idxs.cuda() if data.is_cuda else data_idxs
        hf_data = data[data < self.word_vocab_size]
        hf_data_idxs = data_idxs[data < self.word_vocab_size]
        hf_data = Var(hf_data)
        hf_embeddings = self.ovectors(hf_data)
        if hf_data.size(0) < data.size(0):
            lf_data = data[data >= self.word_vocab_size]
            lf_data_idxs = data_idxs[data >= self.word_vocab_size]
            lf_data = Var(lf_data)
            spelling_data = self.wordidx2spelling(lf_data).data.clone().long()
            spelling = spelling_data[:, :-1]
            lengths = spelling_data[:, -1]
            if self.char_composition == 'RNN':
                lf_embeddings = self.batch_rnn(spelling, lengths,
                                               self.context_rnn,
                                               self.context_char_embedding
                                               )
            elif self.char_composition == 'CNN':
                lf_embeddings = self.batch_cnn(spelling, self.context_char_embedding,
                                               self.context_c1d_3g,
                                               self.context_c1d_4g,
                                               self.context_c1d_5g,
                                               self.context_c1d_6g)
            else:
                raise BaseException("unknown char_composition")

            embeddings = torch.cat([hf_embeddings, lf_embeddings], dim=0)
            f_idx = torch.cat([hf_data_idxs, lf_data_idxs], dim=0)
            embeddings[data_idxs, :] = embeddings[f_idx, :]
            # print('c', hf_data.size(0), lf_data.size(0))
            del spelling_data, spelling, lengths, f_idx, lf_data, lf_data_idxs, lf_embeddings
        else:
            embeddings = hf_embeddings
        embeddings = embeddings.view(bs, cs, self.embedding_size)
        del hf_data, data, data_idxs, hf_data_idxs, hf_embeddings
        return embeddings

    def forward(self, data, is_input=True):
        # data = Var(data, requires_grad=False)  #  we make it a Var just so that it can be used with an embedding layer
        data = data.cuda() if self.wordidx2spelling.weight.is_cuda else data
        if is_input:
            # data = {batch_size, max_seq_len}
            # data_lengths = {batch_size}
            return self.input_vectors(data)
        else:
            # data = {batch_size, 2 x window_size, max_seq_len}
            # data_lengths = {batch_size, 2 x window_size}
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
        self.ivectors.weight = nn.Parameter(FT(
            self.word_vocab_size, self.embedding_size).uniform_(-0.5 / self.embedding_size, 0.5 / self.embedding_size))
        self.ovectors.weight = nn.Parameter(FT(
            self.word_vocab_size, self.embedding_size).uniform_(-0.5 / self.embedding_size, 0.5 / self.embedding_size))
        self.ivectors.weight.requires_grad = True
        self.ovectors.weight.requires_grad = True

    def init_cuda(self,):
        pass

    def forward(self, data, lengths=None, is_input=True):
        if is_input:
            return self.input_vectors(data)
        else:
            return self.context_vectors(data)

    def query(self, word_idx):
        assert isinstance(word_idx, int)
        self.eval()
        word_idx = torch.LongTensor([word_idx])
        if self.ivectors.weight.is_cuda:
            word_idx = word_idx.cuda()
        embedding = self.ivectors(Var(word_idx))
        vecs = embedding.data.cpu().numpy()
        return vecs

    def input_vectors(self, data):
        # data = {batch_size}
        v = Var(LT(data))  # , requires_grad=False)
        v = v.cuda() if self.ivectors.weight.is_cuda else v
        vecs = self.ivectors(v)
        return vecs

    def context_vectors(self, data):
        # data = {batch_size x 2 * window_size}
        v = Var(LT(data))  # , requires_grad=False)
        v = v.cuda() if self.ivectors.weight.is_cuda else v
        vecs = self.ovectors(v)
        return vecs

    def save_model(self, path):
        torch.save(self, path)


class SGNS(nn.Module):
    def __init__(self, embedding_model, num_neg_samples=20, window_size=5, weights=None):
        super(SGNS, self).__init__()
        self.embedding_model = embedding_model
        self.num_neg_samples = num_neg_samples
        self.context_size = window_size * 2
        self.weights = None
        if weights is not None:
            wf = np.power(weights, 0.75)
            wf = wf / wf.sum()
            self.weights = FT(wf)

    def init_cuda(self,):
        self = self.cuda()
        self.embedding_model.init_cuda()

    def sample_noise(self, batch_size):
        if self.weights is not None:
            nwords = torch.multinomial(self.weights, batch_size * self.context_size * self.num_neg_samples,
                                       replacement=True).view(batch_size, -1)
        else:
            nwords = FT(batch_size, self.context_size * self.num_neg_samples).uniform_(
                    0, self.embedding_model.noise_vocab_size - 1).long()
        return nwords

    def forward(self, iword, owords, nwords):
        ivectors = self.embedding_model(iword, is_input=True).unsqueeze(2)
        ovectors = self.embedding_model(owords, is_input=False)
        nvectors = -self.embedding_model(nwords, is_input=False)
        # log_prob of belonging in "true" class
        nll = -nn.functional.logsigmoid(torch.bmm(ovectors, ivectors).squeeze()).mean(1)
        nll_negated_noise = -nn.functional.logsigmoid(torch.bmm(nvectors, ivectors).squeeze()).view(
                -1, self.context_size, self.num_neg_samples).sum(2).mean(1)  # log_prob of "noise" class
        loss = (nll + nll_negated_noise).mean()
        return loss

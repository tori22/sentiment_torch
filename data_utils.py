# -*- coding: utf-8 -*-
# file: data_utils.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

import os
import pickle
import numpy as np
from torch.utils.data import Dataset
import jieba
import sys
jieba.load_userdict('./dict.txt')
def cut(string):
    seg_list = jieba.cut(string, cut_all = False)
    return seg_list

def load_word_vec(path, word2idx=None):
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        if word2idx is None or tokens[0] in word2idx.keys():
        #if tokens[0] in word2idx.keys():
            word_vec[tokens[0]] = np.asarray(tokens[1:])
    return word_vec


def build_embedding_matrix(word2idx, embed_dim, type):
    embedding_matrix_file_name = '{0}_{1}_embedding_matrix.dat'.format(str(embed_dim), type)
    if os.path.exists(embedding_matrix_file_name):
        print('loading embedding_matrix:', embedding_matrix_file_name)
        embedding_matrix = pickle.load(open(embedding_matrix_file_name, 'rb'))
    else:
        print('loading word vectors...')
        embedding_matrix = np.zeros((len(word2idx) + 2, embed_dim))  # idx 0 and len(word2idx)+1 are all-zeros
        fname = '../Tencent_AILab_ChineseEmbedding.txt'
        word_vec = load_word_vec(fname, word2idx=word2idx)
        print('building embedding_matrix:', embedding_matrix_file_name)
        for word, i in word2idx.items():
            vec = word_vec.get(word)
            if vec is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(embedding_matrix_file_name, 'wb'))
    return embedding_matrix


class Tokenizer(object):
    def __init__(self, lower=False, max_seq_len=None, max_aspect_len=None):
        self.lower = lower
        self.max_seq_len = max_seq_len
        self.max_aspect_len = max_aspect_len
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 1

    def fit_on_text(self, text):
        if self.lower:
            text = text.lower()
        words = text.split()
        for word in words:
            cut_list = cut(word.strip())
            for letter in cut_list:
                if letter not in self.word2idx:
                    self.word2idx[letter] = self.idx
                    self.idx2word[self.idx] = letter
                    self.idx += 1

    @staticmethod
    def pad_sequence(sequence, maxlen, dtype='int64', padding='pre', truncating='pre', value=0.):
        x = (np.ones(maxlen) * value).astype(dtype)
        if truncating == 'pre':
            trunc = sequence[-maxlen:]
        else:
            trunc = sequence[:maxlen]
        trunc = np.asarray(trunc, dtype=dtype)
        if padding == 'post':
            x[:len(trunc)] = trunc
        else:
            x[-len(trunc):] = trunc
        return x

    def text_to_sequence(self, text, reverse=False):
        if self.lower:
            text = text.lower()
        words = text.split()
        unknownidx = len(self.word2idx)+1
        #print(self.word2idx)
        #sequence = [self.word2idx[w] if w in self.word2idx else unknownidx for w in word for word in words]
        sequence = []
        for word in words:
            cut_list = cut(word.strip())
            #print(cut_list)
            for w in cut_list:
                if w in self.word2idx:
                    sequence.append(self.word2idx[w])
                else:
                    sequence.append(unknownidx)
        #print(sequence)

        if len(sequence) == 0:
            sequence = [0]
        pad_and_trunc = 'post'  # use post padding together with torch.nn.utils.rnn.pack_padded_sequence
        if reverse:
            sequence = sequence[::-1]
        return Tokenizer.pad_sequence(sequence, self.max_seq_len, dtype='int64', padding=pad_and_trunc, truncating=pad_and_trunc)


class ABSADataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class ABSADatesetReader:
    @staticmethod
    def __read_text__(fnames):
        text = ''
        for fname in fnames:
            fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
            lines = fin.readlines()
            fin.close()
            for i in range(0, len(lines), 3):
                text_left, _, text_right = [s.strip() for s in lines[i].partition("$T$")]
                aspect = lines[i + 1].strip()
                text_raw = text_left + " " + aspect + " " + text_right
                text += text_raw + " "
        return text

    @staticmethod
    def __read_data__(fname, tokenizer):
        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()
        fin.close()

        all_data = []
        for i in range(0, len(lines), 3):
            #text_left, _, text_right = [s.strip() for s in lines[i].partition("$T$")]
            aspect = lines[i + 1].strip()
            polarity = lines[i + 2].strip()
            text_raw_without_aspect_indices = tokenizer.text_to_sequence(lines[i].strip())
            #text_raw_indices = tokenizer.text_to_sequence(text_left + " " + aspect + " " + text_right)
            #text_raw_without_aspect_indices = tokenizer.text_to_sequence(text_left + " " + text_right)
            #text_left_indices = tokenizer.text_to_sequence(text_left)
            #text_left_with_aspect_indices = tokenizer.text_to_sequence(text_left + " " + aspect)
            #text_right_indices = tokenizer.text_to_sequence(text_right, reverse=True)
            #text_right_with_aspect_indices = tokenizer.text_to_sequence(" " + aspect + " " + text_right, reverse=True)
            aspect_indices = tokenizer.text_to_sequence(aspect)
            polarity = int(polarity)+1

            data = {
                'text_raw_indices': text_raw_without_aspect_indices,
                #'text_raw_indices': text_raw_indices,
                #'text_raw_without_aspect_indices': text_raw_without_aspect_indices,
                #'text_left_indices': text_left_indices,
                #'text_left_with_aspect_indices': text_left_with_aspect_indices,
                #'text_right_indices': text_right_indices,
                #'text_right_with_aspect_indices': text_right_with_aspect_indices,
                'aspect_indices': aspect_indices,
                'polarity': polarity,
            }

            all_data.append(data)
        return all_data

    def __init__(self, dataset='twitter', embed_dim=100, max_seq_len=40):
        print("preparing {0} dataset...".format(dataset))
        fname = {
            'ccf': {
                'train':'./datasets/train.txt',
                'test':'./datasets/test.txt',
                'final':'./datasets/final.txt'
                }
        }
        text = ABSADatesetReader.__read_text__([fname[dataset]['train'], fname[dataset]['test'], fname[dataset]['final']])
        tokenizer = Tokenizer(max_seq_len=max_seq_len)
        tokenizer.fit_on_text(text)
        self.embedding_matrix = build_embedding_matrix(tokenizer.word2idx, embed_dim, dataset)
        self.train_data = ABSADataset(ABSADatesetReader.__read_data__(fname[dataset]['train'], tokenizer))
        self.test_data = ABSADataset(ABSADatesetReader.__read_data__(fname[dataset]['test'], tokenizer))
        self.final_data = ABSADataset(ABSADatesetReader.__read_data__(fname[dataset]['final'], tokenizer))

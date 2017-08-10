#!/usr/bin/env python2
# -*- coding: utf-8 -*-

class config(object):

    
    emb_dim = 128
    enco_hdim = 64
    vocab_size = 30000
    beam_size = 3
    batch_size = 32
    

    filter_ngram = 5
    num_filters = 1
    num_layers = 5
    
    rnn_hdim = 256
    doc_len = 400
    sum_len = 100
    max_len = 50
    
    num_epoch = 10
    momentum = 0.95
    lr = 0.01
    max_grad_norm = 10.0
    model_path = "./model" 

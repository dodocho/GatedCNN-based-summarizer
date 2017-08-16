# -*- coding: utf-8 -*-
class config(object):
    
    emb_dim = 128
    enco_hdim = emb_dim
    vocab_size = 30000
    beam_size = 3
    batch_size = 16
    
    filter_ngram = 3
    num_filters = 1
    num_layers = 2
    deco_num_layers = 2
    rnn_hdim = 256
    doc_len = 400
    sum_len = 100
    max_len = 50
    generate_l=20
    
    num_epoch = 10
    momentum = 0.95
    lr = 0.001
    max_grad_norm = 10.0
    model_path = "./model" 
    stddev = 0.5

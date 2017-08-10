#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 16:00:52 2017

@author: zy
"""

import tensorflow as tf
import Config
conf = Config.config()

def inference_encoder(infer_input2id):
    
    #in_channels=1
    
    with tf.device('/cpu:0'), tf.variable_scope("embedding", reuse=True):
        emb_w = tf.get_variable(name="emb_W")
        h = tf.nn.embedding_lookup(emb_w, infer_input2id)
        h = tf.expand_dims(h, 0)
        h = tf.expand_dims(h, -1)
    
    for i in range(conf.num_layers):
        #(filter_w, num_filters) = (1, conf.num_filters) if i<(conf.num_layers-1) else (conf.emb_dim-conf.enco_hdim+1, 1)
        #filter_shape = [conf.filter_ngram, filter_w, in_channels , num_filters]
        
        with tf.variable_scope("layer_%d" % i, reuse=True):
            # Convolution Layer
            W = tf.get_variable(name="w_%d" % i)
            b = tf.get_variable(name="b_%d" % i)
            conv = tf.nn.conv2d(
                h,
                W,
                strides=[1, 1, 1, 1],
                padding="VALID",
                name="conv")
            
            W_gated = tf.get_variable(name="w_gated_%d" % i)
            b_gated = tf.get_variable(name="b_gated_%d" % i)
            conv_gated = tf.nn.conv2d(
                h,
                W_gated,
                strides=[1, 1, 1, 1],
                padding="VALID",
                name="conv_gated")
            
    
            #gated CNN
            h = tf.nn.bias_add(conv, b)*tf.sigmoid(tf.nn.bias_add(conv_gated, b_gated))
        
        #in_channels = conf.num_filters
    
    return h
            

def beam_decoder(enco_h, vocab, session):
    enco_h = enco_h[0,:,:,0]
    reverse_vocab = dict([word, _id] for _id, word in enumerate(vocab))
    
    
    with tf.variable_scope("decode", reuse=True):
        
        wz1 = tf.get_variable(name="wz1") 
        bz1 = tf.get_variable(name="bz1")
        
        wr1 = tf.get_variable(name="wr1") 
        br1 = tf.get_variable(name="br1")
        
        wh1 = tf.get_variable(name="wh1") 
        bh1 = tf.get_variable(name="bh1")
        
        h0 = tf.get_variable(name="h0")
        
        att_w = tf.get_variable(name="att_w") 
        
        softmax_w = tf.get_variable(name="softmax_w") 
        softmax_b = tf.get_variable(name="softmax_b")
        
    
    with tf.device('/cpu:0'), tf.variable_scope("embedding", reuse=True):
        emb_w = tf.get_variable(name="emb_W") 

    
    
    def decode_step(seq, prev_h_state):
        #
        att_temp = tf.matmul(tf.matmul(enco_h, att_w), tf.transpose(prev_h_state,[1,0]))
        att_score = tf.nn.softmax(att_temp, 0)
        att_cont = tf.reduce_sum(enco_h*att_score, 0)
        att_cont = tf.expand_dims(att_cont, 0)
        
        deco_input = tf.nn.embedding_lookup(emb_w, seq[-1][0])
        deco_input = tf.expand_dims(deco_input, 0)
        
        #(emb_dim + enco_hdim + rnn_hdim)
        cat = tf.concat([deco_input, att_cont, prev_h_state], axis=1)
        
        
        #batch_size * rnn_hdim for both
        update = tf.sigmoid(tf.matmul(cat, wz1) + bz1)
        reset = tf.sigmoid(tf.matmul(cat, wr1) + br1)
        
        #batch_size * (emb_dim + enco_hdim + rnn_hdim)
        cat_cadi=tf.concat([deco_input, att_cont, prev_h_state * reset],axis=1)
        
        
        #batch_size * rnn_hdim
        ht_cadidation = tf.tanh(tf.matmul(cat_cadi, wh1) + bh1)
        
        #batch_size * rnn_hdim
        ht = update * prev_h_state + (1-update) * ht_cadidation
        
        pred = tf.matmul(ht, softmax_w) + softmax_b
        
        prob = tf.nn.softmax(pred)
        
        
        return prob[0,:], ht
    ###    
    
    def check_all_stop(seqs):
        for seq in seqs:
            if not seq[-1]:
                return False
        return True
    
    
    def seq_mul(seq):
        out = 1.0
        for x in seq:
            out *= x
        return out
    #seq: [[word,word],[word,word],[word,word]]
    #output: [[word,word,word],[word,word,word],[word,word,word]]
    
    def beam_search_step(top_seqs):
        all_seqs = []
        for seq in top_seqs:
            seq_score = seq_mul([score for _, score, _ in seq])
            if seq[-1][0] == reverse_vocab['<eos>']:
                all_seqs.append((seq, seq_score, True))
                continue
            
            deco_h_state_seqi = seq[-1][-1]
            prob, deco_h_state = decode_step(seq, deco_h_state_seqi)
            prob = session.run(prob)
            ouput_step = [[idx, prob_i] for idx, prob_i in enumerate(prob)]        
            ouput_step = sorted(ouput_step, key=lambda x: x[1], reverse=True)
            for word_prob in ouput_step[:conf.beam_size]:
                word_index = word_prob[0]
                word_score = word_prob[1]
                score = seq_score * word_score
                temp = word_prob + [deco_h_state]
                
                
                rs_seq = seq + [temp]
                stop = (word_index == reverse_vocab['<eos>'])
                all_seqs.append([rs_seq, score, stop])
            
        all_seqs = sorted(all_seqs, key = lambda seq: seq[1], reverse=True)   
        all_stop = check_all_stop(all_seqs[:conf.beam_size])   
        topk_seqs = [seq for seq,_,_ in all_seqs[:conf.beam_size]]
           
        return topk_seqs, all_stop
    
    ####################################
    deco_h_state = tf.expand_dims(h0, 0)
    top_seqs = [[[reverse_vocab['<go>'], 1.0, deco_h_state]]]
    
    generate_l=50
    for l in range(generate_l):
        top_seqs, all_stop = beam_search_step(top_seqs)
        if all_stop=='True':
            break
        print l
    
    return top_seqs
    
def main(infer_input, vocab, sess):
    enco_h_infer = inference_encoder(infer_input)
    
    top_seqs = beam_decoder(enco_h_infer, vocab, sess)
    
    return top_seqs
            
            
            
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
            
            
            
            
            
            
            
            
            
        
    
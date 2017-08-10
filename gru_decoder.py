# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import Config
conf = Config.config()



def gru_decoder(enco_h, mask_h, deco_inputs2id, mask_sum_2dims, y_true2id):
    
    with tf.device('/cpu:0'), tf.variable_scope("embedding", reuse=True):
        emb_w = tf.get_variable(name="emb_W", shape=[conf.vocab_size, conf.emb_dim], initializer=tf.truncated_normal_initializer(stddev=0.5)) 
        deco_emb = tf.nn.embedding_lookup(emb_w, deco_inputs2id)
    
    deco_emb_T = tf.transpose(deco_emb, [1, 0, 2])
    mask_sum_3dims = tf.expand_dims(mask_sum_2dims, -1)
    mask_sum_T = tf.transpose(mask_sum_3dims, [1, 0, 2])
    
    #h_length = 380#enco_h.get_shape().as_list()[1]
    
    with tf.variable_scope("decode"):
        
        gru_shape = [conf.emb_dim+conf.enco_hdim+conf.rnn_hdim, conf.rnn_hdim]
        wz1 = tf.get_variable(name="wz1", shape=gru_shape, initializer=tf.truncated_normal_initializer(stddev=0.5)) 
        bz1 = tf.get_variable(name="bz1", shape=gru_shape[1], initializer=tf.constant_initializer(0.0))
        wr1 = tf.get_variable(name="wr1", shape=gru_shape, initializer=tf.truncated_normal_initializer(stddev=0.5)) 
        br1 = tf.get_variable(name="br1", shape=gru_shape[1], initializer=tf.constant_initializer(0.0))
        wh1 = tf.get_variable(name="wh1", shape=gru_shape, initializer=tf.truncated_normal_initializer(stddev=0.5)) 
        bh1 = tf.get_variable(name="bh1", shape=gru_shape[1], initializer=tf.constant_initializer(0.0))
        
        h0 = tf.get_variable(name="h0", shape=[conf.rnn_hdim], initializer=tf.constant_initializer(0.0))
        h0_repeat = tf.tile([h0],[conf.batch_size, 1])
        
        att_w = tf.get_variable(name="att_w", shape=[conf.enco_hdim, conf.rnn_hdim], initializer=tf.truncated_normal_initializer(stddev=0.5)) 

        softmax_w = tf.get_variable(name="softmax_w", shape=[conf.rnn_hdim, conf.vocab_size],
                                initializer=tf.truncated_normal_initializer(stddev=0.5)) 
        softmax_b = tf.get_variable(name="softmax_b", shape=conf.vocab_size, initializer=tf.constant_initializer(0.0))
    
       
    #     
    def attention(prev_, curr_input):
        #extract 
        curr_enco_h = curr_input[0]
        curr_deco_h = tf.expand_dims(curr_input[1], -1)
        mask_hi = curr_input[2]
        
        temp = tf.matmul(tf.matmul(curr_enco_h, att_w), curr_deco_h)
        temp = tf.add(temp[:, 0], -10000*(1-mask_hi))
        att_score = tf.nn.softmax(temp)
        att_score_2dims = tf.expand_dims(att_score, -1)
        att_context = tf.reduce_sum(curr_enco_h*att_score_2dims, 0)
        return att_context
    
    

    def gru1(prev_h, curr_input):   
        #prev_h: batch_size * rnn_hdim
        
        #batch_size * emb_dim
        deco_input_i = curr_input[0]
        
        #batch_size * enco_hdim
        att_cont = tf.scan(fn = attention,
                           elems = [enco_h, prev_h, mask_h],
                           initializer = tf.constant(np.zeros(conf.enco_hdim), dtype=tf.float32))
        
        #batch_size * 1
        mask_i = curr_input[1]
        
        #batch_size * (emb_dim + enco_hdim + rnn_hdim)
        cat = tf.concat([deco_input_i, att_cont, prev_h], axis=1)
        
        #batch_size * rnn_hdim for both
        update = tf.sigmoid(tf.matmul(cat, wz1) + bz1)
        reset = tf.sigmoid(tf.matmul(cat, wr1) + br1)
        
        #batch_size * (emb_dim + enco_hdim + rnn_hdim)
        cat_cadi=tf.concat([deco_input_i, att_cont, prev_h * reset],axis=1)
        
        #batch_size * rnn_hdim
        ht_cadidation = tf.tanh(tf.matmul(cat_cadi, wh1) + bh1)
        
        #batch_size * rnn_hdim
        ht = update * prev_h + (1-update) * ht_cadidation
        
        #batch_size * rnn_hdim
        ht = mask_i * ht + (1-mask_i) * prev_h
        
        return ht
    
    #0 
    gru_h_T = tf.scan(fn = gru1,
                         elems = [deco_emb_T, mask_sum_T],
                         initializer = h0_repeat)
    #1
    gru_h = tf.transpose(gru_h_T, [1,0,2])
    
    #2
    gru_h_mask = gru_h*mask_sum_3dims
    
    #3
    gru_h_mask_flat = tf.reshape(gru_h_mask, [-1,conf.rnn_hdim])
    
    #4
    pred = tf.matmul(gru_h_mask_flat, softmax_w) + softmax_b
    
    #5
    mask_sum_flat = tf.reshape(mask_sum_2dims, [-1, 1])  
    
    #6
    pred_mask_softmax = tf.nn.softmax(pred*mask_sum_flat)
    
    #7
    y_true2id_flat = tf.reshape(y_true2id, [-1])  
    
    #8
    total_loss=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred_mask_softmax, labels=y_true2id_flat)
    
    #9
    loss = tf.div(tf.reduce_sum(total_loss * mask_sum_flat[:,0]), (tf.reduce_sum(mask_sum_flat)))

    return loss
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
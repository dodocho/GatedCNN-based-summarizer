# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import config
conf = config.config()

def ini_param(var_name, shape, ini_name):
    if ini_name == "norm": initializer=tf.truncated_normal_initializer(stddev=np.sqrt(1/float(shape[0])))#GNN for SC used it
    if ini_name == "glorot_uniform": initializer=tf.contrib.layers.xavier_initializer(uniform=True)#sqrt(6/(in+out))
    if ini_name == "glorot_normal": initializer=tf.contrib.layers.xavier_initializer(uniform=False)#sqrt(3/(in+out))
    if ini_name == "constant0": initializer=tf.constant_initializer(0.0)
    if ini_name == "constant": initializer=tf.constant_initializer(0.001)
    if ini_name == "ortho": initializer=tf.orthogonal_initializer()
    return tf.get_variable(name=var_name, shape=shape, initializer=initializer)

def gru_decoder(enco_h, mask_h, deco_inputs2id, mask_sum_2dims, y_true2id):
    with tf.device('/cpu:0'), tf.variable_scope("embedding", reuse=True):
        emb_w = tf.get_variable(name="emb_W") 
        deco_emb = tf.nn.embedding_lookup(emb_w, deco_inputs2id)
    
    deco_emb_T = tf.transpose(deco_emb, [1, 0, 2])
    mask_sum_3dims = tf.expand_dims(mask_sum_2dims, -1)
    mask_sum_3dims_T = tf.transpose(mask_sum_3dims, [1, 0, 2])
    
    with tf.variable_scope("decode"):
        wz = ini_param("wz", [conf.emb_dim+conf.enco_hdim, conf.rnn_hdim], "ortho")
        uz = ini_param("uz", [conf.rnn_hdim, conf.rnn_hdim], "ortho")
        #bz = ini_param("bz", conf.rnn_hdim, "constant")
        
        wr = ini_param("wr", [conf.emb_dim+conf.enco_hdim, conf.rnn_hdim], "ortho")
        ur = ini_param("ur", [conf.rnn_hdim, conf.rnn_hdim], "ortho")
        #br = ini_param("br", shape=conf.rnn_hdim, "constant")
        
        wh = ini_param("wh", [conf.emb_dim+conf.enco_hdim, conf.rnn_hdim], "ortho")
        uh = ini_param("uh", [conf.rnn_hdim, conf.rnn_hdim], "ortho")
        #bh = ini_param("br", shape=conf.rnn_hdim, "constant")
        
        h0 = ini_param("h0", [conf.rnn_hdim], "constant0")
        h0_repeat = tf.tile([h0], [conf.batch_size, 1])#batch_size * rnn_hdim
        
        #att_w = tf.get_variable(name="att_w", shape=[conf.enco_hdim, conf.rnn_hdim], initializer=tf.truncated_normal_initializer(stddev=0.5)) 
        w_att = ini_param("w_att", [conf.enco_hdim+conf.rnn_hdim, conf.enco_hdim], "norm")
        v_att = ini_param("v_att", [conf.enco_hdim, 1], "norm")
        
        softmax_w = ini_param("softmax_w", [conf.rnn_hdim, conf.vocab_size], "norm")
        softmax_b = ini_param("softmax_b", conf.vocab_size, "constant0")
    
    def attention(prev_, curr_input):
        enco_hi = curr_input[0]#[doc_len, enco_hdim]
        mask_hi = curr_input[1]#[doc_len, ]
        gru_hi  = curr_input[2]#[rnn_hdim, ]
        
        gru_hi_repeat = tf.tile([gru_hi], [conf.doc_len, 1])#[doc_len, rnn_hdim]
        cat = tf.concat([enco_hi, gru_hi_repeat], axis=1)#[doc_len, enco_hdim+rnn_hdim]
        
        att = tf.matmul(tf.tanh(tf.matmul(cat, w_att)), v_att)#[doc_len, 1]
        att = tf.add(att[:, 0], -100000*(1-mask_hi))
        att_score = tf.nn.softmax(att)
        att_score_2dims = tf.expand_dims(att_score, -1)
        att_context = tf.reduce_sum(enco_hi*att_score_2dims, 0)
        return att_context
        
    def gru_layer(prev_h, curr_input_i):
        deco_input_i = curr_input_i[0]#[batch_size, emb_dim]
        mask_i = curr_input_i[1]#[batch_size, emb_dim]
        
        #att_cont: [batch_size, enco_hdim] 
        att_cont = tf.scan(fn = attention,
                           #enco_h: [batch_size, doc_len, enco_hdim]
                           #mask_h: [batch_size, doc_len]
                           #prev_h: [batch_size, rnn_hdim]
                           elems = [enco_h, mask_h, prev_h],
                           initializer = tf.constant(np.zeros(conf.enco_hdim), dtype=tf.float32))
        
        #x_cat: [batch_size, emb_dim+enco_hdim] 
        x_cat = tf.concat([deco_input_i, att_cont], axis=1)
        
        # both update and reset: [batch_size, rnn_hdim]
        update = tf.sigmoid(tf.matmul(x_cat, wz) + tf.matmul(prev_h, uz))
        reset = tf.sigmoid(tf.matmul(x_cat, wr) + tf.matmul(prev_h, ur))
        
        #h_cadi: [batch_size, rnn_hdim]
        h_cadi = tf.tanh(tf.matmul(x_cat, wh) + tf.matmul(prev_h * reset, uh))
        
        #ht [batch_size * rnn_hdim
        ht = update * prev_h + (1-update) * h_cadi
        
        #batch_size * rnn_hdim
        ht = mask_i * ht + (1-mask_i) * prev_h
        
        return ht

    #both deco_emb_T, mask_sum_T: [sum_len, batch_size, emb_dim]
    gru_h_T = tf.scan(fn = gru_layer,
                      elems = [deco_emb_T, mask_sum_3dims_T],
                      initializer = h0_repeat)
    
    gru_h = tf.transpose(gru_h_T, [1,0,2])
    gru_h_mask = gru_h*mask_sum_3dims
    gru_h_mask_flat = tf.reshape(gru_h_mask, [-1, conf.rnn_hdim])
    pred = tf.matmul(gru_h_mask_flat, softmax_w) + softmax_b

    mask_sum_flat = tf.reshape(mask_sum_2dims, [-1, 1])  
    pred_mask_softmax = tf.nn.softmax(pred*mask_sum_flat)
    y_true2id_flat = tf.reshape(y_true2id, [-1])  
    total_loss=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred_mask_softmax, labels=y_true2id_flat)
    loss = tf.div(tf.reduce_sum(total_loss * mask_sum_flat[:,0]), (tf.reduce_sum(mask_sum_flat)))
    
    return loss

    

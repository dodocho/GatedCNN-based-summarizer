# -*- coding: utf-8 -*-
import tensorflow as tf
import config
conf = config.config()

#rebuild computational graph
#rebuild encoder
def inference_encoder(infer_input2id):
    with tf.device('/cpu:0'), tf.variable_scope("embedding", reuse=True):
        emb_w = tf.get_variable(name="emb_W")
        h = tf.nn.embedding_lookup(emb_w, infer_input2id)
        h = tf.expand_dims(h, 0)
        h = tf.expand_dims(h, -1)
    
    for i in range(conf.num_layers):
        '''
        filter_w = conf.emb_dim #if i<(conf.num_layers-1) else conf.emb_dim-conf.enco_hdim+1
        in_channels = 1
        num_filters = conf.num_filters if i<(conf.num_layers-1) else 1
        filter_shape = [conf.filter_ngram, filter_w, in_channels , num_filters]
        '''
        with tf.variable_scope("layer_%d" % i, reuse=True):
            # Convolution Layer
            W = tf.get_variable(name="w_%d" % i)
            b = tf.get_variable(name="b_%d" % i)
            conv = tf.nn.conv2d(
                h,
                W,
                strides=[1, 1, 1, 1],
                padding="SAME",
                name="conv")
            
            W_gated = tf.get_variable(name="w_gated_%d" % i)
            b_gated = tf.get_variable(name="b_gated_%d" % i)
            conv_gated = tf.nn.conv2d(
                h,
                W_gated,
                strides=[1, 1, 1, 1],
                padding="SAME",
                name="conv_gated")
            #gated CNN
            h = tf.nn.bias_add(conv, b)*tf.sigmoid(tf.nn.bias_add(conv_gated, b_gated))
    
    return h
            

def beam_decoder(enco_h, vocab, session):
    enco_h = enco_h[0,:,:,0]
    reverse_vocab = dict([word, _id] for _id, word in enumerate(vocab))
    
    with tf.variable_scope("decode", reuse=True):
        wz = tf.get_variable(name="wz") 
        uz = tf.get_variable(name="uz")
        wr = tf.get_variable(name="wr") 
        ur = tf.get_variable(name="ur")
        wh = tf.get_variable(name="wh") 
        uh = tf.get_variable(name="uh")
        h0 = tf.get_variable(name="h0")
        w_att = tf.get_variable(name="w_att") 
        v_att = tf.get_variable(name="v_att") 
        softmax_w = tf.get_variable(name="softmax_w") 
        softmax_b = tf.get_variable(name="softmax_b")
        
    with tf.device('/cpu:0'), tf.variable_scope("embedding", reuse=True):
        emb_w = tf.get_variable(name="emb_W") 

    def decode_step(seq, prev_h):
        gru_hi_repeat = tf.tile(prev_h, [enco_h.get_shape().as_list()[0], 1])
        cat = tf.concat([enco_h, gru_hi_repeat], axis=1)#[doc_len, enco_hdim+rnn_hdim]
        
        att = tf.matmul(tf.tanh(tf.matmul(cat, w_att)), v_att)#[doc_len, 1]
        att_score = tf.nn.softmax(att)
        #att_score_2dims = tf.expand_dims(att_score, -1)
        att_cont = tf.reduce_sum(enco_h*att_score, 0)#[1, enco_hdim]
        att_cont = tf.expand_dims(att_cont, 0)
        
        deco_input_i = tf.nn.embedding_lookup(emb_w, seq[-1][0])
        deco_input_i = tf.expand_dims(deco_input_i, 0)#[1, emb_dim]
        
        #emb_dim + enco_hdim
        x_cat = tf.concat([deco_input_i, att_cont], axis=1)#[1, emb_dim + enco_hdim]
        
        # both update and reset: [batch_size, rnn_hdim]
        update = tf.sigmoid(tf.matmul(x_cat, wz) + tf.matmul(prev_h, uz))
        reset = tf.sigmoid(tf.matmul(x_cat, wr) + tf.matmul(prev_h, ur))
    
        #h_cadi: [batch_size, rnn_hdim]
        h_cadi = tf.tanh(tf.matmul(x_cat, wh) + tf.matmul(prev_h * reset, uh))
        
        #ht [batch_size * rnn_hdim
        ht = update * prev_h + (1-update) * h_cadi
        
        pred = tf.matmul(ht, softmax_w) + softmax_b
        prob = tf.nn.softmax(pred)
   
        return prob[0,:], ht

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
   
    for l in range(conf.generate_l):
        top_seqs, all_stop = beam_search_step(top_seqs)
        if all_stop=='True':
            break
        print l
    
    return top_seqs
    
def main(infer_input, vocab, sess):
    enco_h_infer = inference_encoder(infer_input)
    top_seqs = beam_decoder(enco_h_infer, vocab, sess)
    return top_seqs
 

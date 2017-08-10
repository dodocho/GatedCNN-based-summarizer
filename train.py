#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 13:10:53 2017

@author: zy
"""
import tensorflow as tf
import numpy as np
import data_batch
import Config
conf = Config.config()
config_tf = tf.ConfigProto()
config_tf.gpu_options.allow_growth = True
#config_tf.inter_op_parallelism_threads = 1
#config_tf.intra_op_parallelism_threads = 1

#placeholder for input
tf_input_x = tf.placeholder(tf.int32, [conf.batch_size, conf.doc_len], name="input_x")
tf_y_true = tf.placeholder(tf.int32, [conf.batch_size, conf.sum_len], name="y_true")
tf_deco_input = tf.placeholder(tf.int32, [conf.batch_size, conf.sum_len], name="deco_input")
tf_doc_mask_4dims = tf.placeholder(tf.float32, [conf.batch_size, conf.doc_len, conf.emb_dim, 1], name="doc_mask_4dims")
tf_sum_mask_2dims = tf.placeholder(tf.float32, [conf.batch_size, conf.sum_len], name="sum_mask_2dims")

#train_doc2id, train_y_true2id, train_deco_inputs2id, doc_mask_2dims, sum_mask_2dims, vocab = data_batch.data_batch()
dir_cpu ="/home/yang/IJCNLP_2017/code/train_id_data/"

train_doc2id = np.loadtxt(dir_cpu+"train_doc2id.txt").astype('int32')
train_deco_inputs2id = np.loadtxt(dir_cpu+"train_deco_inputs2id.txt").astype('int32')
train_y_true2id = np.loadtxt(dir_cpu+"train_y_true2id.txt").astype('int32')
doc_mask_2dims = list(np.loadtxt(dir_cpu+"doc_mask_2dims.txt").astype('int32'))
sum_mask_2dims = list(np.loadtxt(dir_cpu+"sum_mask_2dims.txt").astype('int32'))
vocab = open(dir_cpu+'vocab.txt', 'r').read().split()

dims_expanded = np.array([np.zeros((conf.emb_dim,1)), np.ones((conf.emb_dim,1))])
conf.vocab_size = len(vocab)
iterations = len(train_doc2id)/conf.batch_size

#Computational graph
import cnn_encoder
import gru_decoder

enco_h, mask_h = cnn_encoder.cnn_encoder(tf_input_x, tf_doc_mask_4dims)
loss = gru_decoder.gru_decoder(enco_h, mask_h, tf_deco_input, tf_sum_mask_2dims, tf_y_true)

#Adam Optimizer with gradients clip
tvars = tf.trainable_variables()
grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), conf.max_grad_norm)
optimizer = tf.train.AdamOptimizer(conf.lr)
optimizer_tf = optimizer.apply_gradients(zip(grads, tvars))


    
with tf.Session(config=config_tf) as sess: 
    tf.global_variables_initializer().run()
    for i in range(conf.num_epoch):
        
        train_doc2id, train_y_true2id, train_deco_inputs2id, doc_mask_2dims, sum_mask_2dims = \
            data_batch.batch_shuffle(train_doc2id, train_y_true2id, train_deco_inputs2id, doc_mask_2dims, sum_mask_2dims)
        
        doc_mask_4dims = dims_expanded[doc_mask_2dims]
        for k in range(0, iterations*conf.batch_size, conf.batch_size):
            if k!=0 or i!=0:
                tf.get_variable_scope().reuse_variables()
            l =sess.run([loss, enco_h, optimizer_tf], feed_dict={tf_input_x: train_doc2id[k:k+conf.batch_size],
                                           tf_y_true: train_y_true2id[k:k+conf.batch_size],
                                           tf_deco_input: train_deco_inputs2id[k:k+conf.batch_size],
                                           tf_doc_mask_4dims:doc_mask_4dims[k:k+conf.batch_size],
                                           tf_sum_mask_2dims:sum_mask_2dims[k:k+conf.batch_size]})
    
            
            print "epoch:"+str(i)+" "+"iteration:"+str(k/conf.batch_size)
            print 'enco_h:'
            print l[1][0]
            
            print l[0]
            var = tf.trainable_variables()
            
            print var[0].name
            print sess.run(var[0][0])
            print '\n'
            print var[-2].name
            print sess.run(var[-2][0])
            print '\n'
            
       
        model_saver = tf.train.Saver()
        model_saver.save(sess,"model_e%d_i%d"%(i,k))

        import beam_search
        with tf.Session() as sess1:
            model_saver = tf.train.Saver()
            model_saver.restore(sess1, "model_e%d_i%d"%(i,k))
            beam_seqs = beam_search.main(train_doc2id[0], vocab, sess1)    
            
        for seq in beam_seqs:
            temp=[]
            for word in seq:
                temp.append(vocab[word[0]])
            print " ".join(temp)
            print "\n"  
        

        
                        
                        
                        
                        
                        
                        
                        
                        


    

    
    
    

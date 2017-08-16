import tensorflow as tf
import config
conf = config.config()

def cnn_encoder(input_x, mask_4dims):
    with tf.device('/cpu:0'), tf.variable_scope("embedding"):
        emb_w = tf.get_variable(name="emb_W", shape=[conf.vocab_size, conf.emb_dim], initializer=tf.truncated_normal_initializer(stddev=0.5))
        h = tf.nn.embedding_lookup(emb_w, input_x)
        h = tf.expand_dims(h, -1)

    for i in range(conf.num_layers):
        filter_w = conf.emb_dim #if i<(conf.num_layers-1) else conf.emb_dim-conf.enco_hdim+1
        in_channels = 1
        num_filters = conf.num_filters if i<(conf.num_layers-1) else 1
        filter_shape = [conf.filter_ngram, filter_w, in_channels , num_filters]
    
        with tf.variable_scope("layer_%d" % i):
            #convolution Layer
            W = tf.get_variable(name="w_%d" % i, shape=filter_shape, initializer=tf.truncated_normal_initializer(stddev=0.5))
            b = tf.get_variable(name="b_%d" % i, shape=[num_filters], initializer=tf.constant_initializer(0.0))
            conv = tf.nn.conv2d(
                h,
                W,
                strides=[1, 1, 1, 1],
                padding="SAME",
                name="conv")
            
            W_gated = tf.get_variable(name="w_gated_%d" % i, shape=filter_shape, initializer=tf.truncated_normal_initializer(stddev=0.5))
            b_gated = tf.get_variable(name="b_gated_%d" % i, shape=[num_filters], initializer=tf.constant_initializer(0.0))
            conv_gated = tf.nn.conv2d(
                h,
                W_gated,
                strides=[1, 1, 1, 1],
                padding="SAME",
                name="conv_gated")
            
            h = tf.nn.bias_add(conv, b)*tf.sigmoid(tf.nn.bias_add(conv_gated, b_gated))
    return h
'''
import numpy as np
import tensorflow as tf
h=np.array([[[1,2,3,4],[5,6,7,8],[9,10,11,12]],[[11,21,31,41],[51,61,71,81],[91,101,111,121]]]).astype("float32")
h=np.reshape(h, [2,3,4,1])

filter_shape = [2, 4, 1, 1]
w_con = tf.get_variable(name="w_con", shape=filter_shape, initializer=tf.constant_initializer(1.0))
conv = tf.nn.conv2d(
        h,
        w_con,
        strides=[1, 1, 1, 1],
        padding="SAME",
        name="conv") 

with tf.Session() as sess: 
    tf.global_variables_initializer().run()
    print sess.run(w_con)
    print '\n'
    print sess.run(conv)
'''

















#Gated CNN as encoder

import tensorflow as tf
import Config
conf = Config.config()

def cnn_encoder(input_x, mask_4dims): 
    in_channels=1
    
    with tf.device('/cpu:0'), tf.variable_scope("embedding"):
        #tf.get_variable_scope().reuse_variables()
        emb_w = tf.get_variable(name="emb_W", shape=[conf.vocab_size, conf.emb_dim], initializer=tf.truncated_normal_initializer(stddev=0.5))
        h = tf.nn.embedding_lookup(emb_w, input_x)
        h = tf.expand_dims(h, -1)
        
    for i in range(conf.num_layers):
        (filter_w, num_filters) = (1, conf.num_filters) if i<(conf.num_layers-1) else (conf.emb_dim-conf.enco_hdim+1, 1)
        filter_shape = [conf.filter_ngram, filter_w, in_channels , num_filters]
        
        with tf.variable_scope("layer_%d" % i):
            # Convolution Layer
            W = tf.get_variable(name="w_%d" % i, shape=filter_shape, initializer=tf.truncated_normal_initializer(stddev=0.5))
            b = tf.get_variable(name="b_%d" % i, shape=[num_filters], initializer=tf.constant_initializer(0.0))
            conv = tf.nn.conv2d(
                h,
                W,
                strides=[1, 1, 1, 1],
                padding="VALID",
                name="conv")
            
            W_gated = tf.get_variable(name="w_gated_%d" % i, shape=filter_shape, initializer=tf.truncated_normal_initializer(stddev=0.5))
            b_gated = tf.get_variable(name="b_gated_%d" % i, shape=[num_filters], initializer=tf.constant_initializer(0.0))
            conv_gated = tf.nn.conv2d(
                h,
                W_gated,
                strides=[1, 1, 1, 1],
                padding="VALID",
                name="conv_gated")
            
    
            #gated CNN
            h = tf.nn.bias_add(conv, b)*tf.sigmoid(tf.nn.bias_add(conv_gated, b_gated))
        
        in_channels = conf.num_filters
        h_shape=h.get_shape().as_list()
        
        mask_4dims = mask_4dims[:, (conf.filter_ngram-1):, 0:h_shape[2], :]
        h = h*mask_4dims
        
        in_channels = conf.num_filters
        
    return h[:,:,:,0], mask_4dims[:,:,0,0]


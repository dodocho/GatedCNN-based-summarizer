# -*- coding: utf-8 -*-
import os
import numpy as np
import config
import random
conf = config.config()

def create_data(data_dir, random_dir):
    with open(random_dir, 'r') as f_random:
        number=set(f_random.read().split())
    f_random.close()
    
    docs=[]
    sums=[]
    
    for each_file in os.listdir(data_dir):
        if each_file[14:] in number:
            with open(os.path.join(data_dir, each_file), 'r') as doc:
                f=doc.read().split('\n\n')
                docs.append(f[0].split())
                sums.append(f[1].split())
    
    return docs, sums

def vocab_build(train_doc, vocab_size):
    data=[]
    for doc in train_doc:
        for word in doc:
            data.append(word) 
    
    import collections
    item_freq=collections.Counter(data).most_common(vocab_size)
    vocab=[]
    for word_freq in item_freq:
        vocab.append(word_freq[0])
    
    return vocab
    
def data_generation():
    pub_dir = '/Users/zy/Desktop'
    
    train_dir=pub_dir+'/IJCNLP_2017/training'
    val_dir=pub_dir+'/IJCNLP_2017/val'
    test_dir=pub_dir+'/IJCNLP_2017/test'
    
    train_random=pub_dir+'/IJCNLP_2017/random_sample_for_train.txt'
    val_random=pub_dir+'/IJCNLP_2017/random_sample_for_val.txt'
    test_random=pub_dir+'/IJCNLP_2017/random_sample_for_test.txt'
    
    train_doc, train_sum=create_data(train_dir, train_random)
    val_doc, val_sum=create_data(val_dir, val_random)
    test_doc, test_sum=create_data(test_dir, test_random)
    
    vocab = vocab_build(train_doc, conf.vocab_size-4)
    vocab.append('<unk>')
    vocab.append('<go>')
    vocab.append('<eos>')
    vocab.append('<pad>')
    
    return (train_doc, train_sum), (val_doc, val_sum), (test_doc, test_sum), vocab


def np_int(data):
    return np.array(data).astype('int32')    

    
def word2id(docu, summ, vocab, is_training=False):
    vocab_set = set(vocab)
    docu1=[]
    docs_mask=[]
    for d in docu:
        d_temp = d[:conf.doc_len]
        if is_training:
            d_mask = np.array([1]*conf.doc_len).astype('int32')
            d_mask[len(d_temp):] = 0
            docs_mask.append(d_mask)
            #pad if necessary 
            d_temp+= ['<pad>']*(conf.doc_len-len(d_temp))
        d1 = []
        for word in d_temp:
            if word in vocab_set:
                d1.append(vocab.index(word))
            else:
                d1.append(vocab.index('<unk>'))
        docu1.append(d1)

    if is_training:
        y_true_labels=[]
        deco_inputs=[]
        summ_mask=[]
        y_true1 = []
        deco_input1 = []
        for s in summ:
            y_true = s[:conf.sum_len-1]+['<eos>']
            deco_input = ['<go>']+s[:conf.sum_len-1]
        
            s_mask = np.array([1]*conf.sum_len).astype('int32')
            s_mask[len(y_true):] = 0
            summ_mask.append(s_mask)
            
            y_true+= ['<pad>']*(conf.sum_len-len(y_true))
            deco_input+= ['<pad>']*(conf.sum_len-len(deco_input))
        
            y_true1 = []
            deco_input1 = []
            for w1, w2 in zip(y_true, deco_input):
                if w1 in vocab_set:
                    y_true1.append(vocab.index(w1))
                else:
                    y_true1.append(vocab.index('<unk>'))
                
                if w2 in vocab_set:
                    deco_input1.append(vocab.index(w2))
                else:
                    deco_input1.append(vocab.index('<unk>'))
            
            y_true_labels.append(y_true1)    
            deco_inputs.append(deco_input1)                    
        
        return np_int(docu1), np_int(deco_inputs), np_int(y_true_labels), docs_mask, summ_mask
    
def batch_shuffle(train_doc2id, train_y_true2id, train_deco_inputs2id, doc_mask, sum_mask):
    l = len(train_doc2id)
    data_shuffle=[]
    for i in range(l):
        data_shuffle.append([train_doc2id[i], train_y_true2id[i], train_deco_inputs2id[i], doc_mask[i], sum_mask[i]])
    random.shuffle(data_shuffle)
    
    train_doc2id1=[]
    train_y_true2id1=[]
    train_deco_inputs2id1=[]
    doc_mask1=[]
    sum_mask1=[]
    
    for each_tuple in data_shuffle:
        train_doc2id1.append(each_tuple[0])
        train_y_true2id1.append(each_tuple[1])
        train_deco_inputs2id1.append(each_tuple[2])
        doc_mask1.append(each_tuple[3])
        sum_mask1.append(each_tuple[4])
        
    return np_int(train_doc2id1), np_int(train_y_true2id1), np_int(train_deco_inputs2id1), np_int(doc_mask1), np.array(sum_mask1).astype("float32")

def data_batch():
    print "data_batch..."
    (train_doc, train_sum), (val_doc, val_sum), (test_doc, test_sum), vocab = data_generation()
    print "finished!"
    
    print "convert to id..."           
    train_doc2id, train_deco_inputs2id, train_y_true2id, doc_mask, sum_mask = word2id(train_doc, train_sum, vocab, is_training=True)
    #val_doc2id, val_sum2id = word2id(val_doc, val_sum, vocab)
    #test_doc2id, test_sum2id = word2id(test_doc, test_sum, vocab)
    print "finished!"  
    
    print "shuffle..."        
    train_doc2id, train_y_true2id, train_deco_inputs2id, doc_mask, sum_mask = batch_shuffle(train_doc2id, train_y_true2id, train_deco_inputs2id, doc_mask, sum_mask)
    print "finished!"
    
    return train_doc2id, train_y_true2id, train_deco_inputs2id, doc_mask, sum_mask, vocab
   
  

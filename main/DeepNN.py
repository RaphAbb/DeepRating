"""
Author: Raphael Abbou
Version: python3
"""

import math
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict


####Import Project Python Files
import data_processor
import input_transco



x = tf.placeholder(tf.float64, name = 'x')

def cost(logits, labels):
    """
    Computes the cost using the sigmoid cross entropy
    
    Arguments:
    logits -- vector containing z, output of the last linear unit (before the final sigmoid activation)
    labels -- vector of labels y (1 or 0) 
    
    Note: What we've been calling "z" and "y" in this class are respectively called "logits" and "labels" 
    in the TensorFlow documentation. So logits will feed into z, and labels into y. 
    
    Returns:
    cost -- runs the session of the cost (formula (2))
    """
    
    z = tf.placeholder(tf.float32, name = 'logits')
    y = tf.placeholder(tf.float32, name ='labels')

    cost = tf.nn.sigmoid_cross_entropy_with_logits(logits = z, labels = y)

    sess = tf.Session()
    
    cost = sess.run(cost, feed_dict = {z: logits, y: labels})
    
    sess.close()
    
    return cost

def one_hot_matrix(labels, C):
    """
    Creates a matrix where the i-th row corresponds to the ith class number and the jth column
                     corresponds to the jth training example. So if example j had a label i. Then entry (i,j) 
                     will be 1. 
                     
    Arguments:
    labels -- vector containing the labels 
    C -- number of classes, the depth of the one hot dimension
    
    Returns: 
    one_hot -- one hot matrix
    """
    
    C = tf.constant( C, name = 'C')

    one_hot_matrix = tf.one_hot(labels, C, axis = 0)
    
    sess = tf.Session()

    one_hot = sess.run(one_hot_matrix)

    sess.close()

    return one_hot


if __name__ == "__main__":
    
    orig_col = ['fico','dt_first_pi','flag_fthb','dt_matr','cd_msa',"mi_pct",'cnt_units','occpy_sts',\
              'cltv','dti','orig_upb','ltv','int_rt','channel','ppmt_pnlty','prod_type','st', \
              'prop_type','zipcode','loan_purpose', 'orig_loan_term','cnt_borr','seller_name'\
              ,'servicer_name', 'flag_sc']
    orig_data = pd.read_csv('sample_orig_2016.txt', header = None, sep = '|', index_col = 19)
    orig_data.columns = orig_col
    
    #Transforming string values to Numerical Values
    string_labels = ['flag_fthb','occpy_sts','channel','ppmt_pnlty','prod_type','st', \
                  'prop_type','loan_purpose','seller_name','servicer_name', 'flag_sc']
    X_train = input_transco.label_to_num(orig_data, string_labels)
    X_train.fillna(0)
    
    #Getting the ouput for the Training Set
    mth_data = pd.read_csv('sample_svcg_2016.txt', header = None, sep = '|')
    Y_train = data_processor.get_training_output(mth_data)
    Y_train = Y_train.reindex(X_train.index)
    
    
    

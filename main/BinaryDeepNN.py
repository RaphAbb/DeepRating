"""
Author: Raphael Abbou
Version: python3
DeepNN Code is inspired by Andrew Ng's course Notebooks
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict
import seaborn as sn

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

def ones(shape):
    """
    Creates an array of ones of dimension shape
    
    Arguments:
    shape -- shape of the array you want to create
        
    Returns: 
    ones -- array containing only ones
    """
    
    ones = tf.ones(shape)
    
    sess = tf.Session()

    ones = sess.run(ones)

    sess.close()

    return ones

def create_placeholders(n_x, n_y):
    """
    Creates the placeholders for the tensorflow session.
    
    Arguments:
    n_x -- scalar, number of criterias
    n_y -- scalar, number of classes
    
    Returns:
    X -- placeholder for the data input, of shape [n_x, None] and dtype "float"
    Y -- placeholder for the input labels, of shape [n_y, None] and dtype "float"
    
    - We use None because it let's us be flexible on the number of examples you will for the placeholders.
    """

    X = tf.placeholder(tf.float32, [n_x, None])
    Y = tf.placeholder(tf.float32, [n_y, None])
    
    return X, Y

def initialize_parameters(n_x, n_y):
    """
    Initializes parameters to build a neural network with tensorflow.
    
    Returns:
    parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3
    """
        
    W1 = tf.get_variable("W1", [18,n_x], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b1 = tf.get_variable("b1", [18,1], initializer = tf.zeros_initializer())
    W2 = tf.get_variable("W2", [12,18], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b2 = tf.get_variable("b2", [12,1], initializer = tf.zeros_initializer())
    W3 = tf.get_variable("W3", [n_y,12], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b3 = tf.get_variable("b3", [n_y,1], initializer = tf.zeros_initializer())

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}
    
    return parameters

def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX
    
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """
    
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    
    Z1 = tf.add(tf.matmul(W1,X), b1)
    A1 = tf.nn.relu(Z1)
    Z2 = tf.add(tf.matmul(W2,A1), b2)
    A2 = tf.nn.relu(Z2)
    Z3 = tf.add(tf.matmul(W3,A2), b3)
    
    return Z3


def compute_cost(Z3, Y):
    """
    Computes the cost
    
    Arguments:
    Z3 -- output of forward propagation (output of the last LINEAR unit)
    Y -- "true" labels vector placeholder, same shape as Z3
    
    Returns:
    cost - Tensor of the cost function
    """
    
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)
    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = logits, labels = labels))
    
    
    return cost

def model(X_train, Y_train, X_test, Y_test, costs, learning_rate = 0.001,#learning_rate = 0.0001,
          #num_epochs = 1500, minibatch_size = 32, print_cost = True):
          num_epochs = 100, minibatch_size = 32, print_cost = True):
    """
    A three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.
    
    Arguments:
    X_train -- training set
    Y_train -- test set
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    
    ops.reset_default_graph()
    (n_x, m) = X_train.shape
    n_y = Y_train.shape[0]
    
    X, Y = create_placeholders(n_x, n_y)

    parameters = initialize_parameters(n_x, n_y)
    
    Z3 = forward_propagation(X, parameters)
    
    cost = compute_cost(Z3, Y)

    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        
        sess.run(init)
        
        for epoch in range(num_epochs):

            epoch_cost = 0.
            num_minibatches = int(m / minibatch_size)
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size)

            for minibatch in minibatches:

                (minibatch_X, minibatch_Y) = minibatch
                
                _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                
                epoch_cost += minibatch_cost / num_minibatches

            # Print the cost every epoch
            if print_cost == True and epoch % 100 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)
                
        # plot the cost
        #plt.plot(np.squeeze(costs[1:]))
        plt.plot(np.squeeze(costs))
        
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print ("Parameters have been trained!")

        # Calculate the correct predictions
        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))
        #correct_prediction = tf.equal(tf.argmax(Z3[1:,]), tf.argmax(Y[1:,]))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
        
        
        confusion = tf.confusion_matrix(labels=tf.argmax(Y), predictions=tf.argmax(Z3), num_classes=2)
        print("Train Confusion Matrix:")
        print(pd.DataFrame(confusion.eval({X: X_train, Y: Y_train}), index = ['Actual Survival', 'Actual Default'], columns = ['Predicted Survival', 'Predicted Default']))
        print("Test Confusion Matrix:")
        print(pd.DataFrame(confusion.eval({X: X_test, Y: Y_test}), index = ['Actual Survival', 'Actual Default'], columns = ['Predicted Survival', 'Predicted Default']))
        
#        auc = tf.metrics.auc(labels=tf.argmax(Y), predictions=tf.argmax(Z3))
#        print ("Train AUC:", auc.eval({X: X_train, Y: Y_train}))
#        print ("Test AUC:", auc.eval({X: X_test, Y: Y_test}))
        
        
        
#        df = pd.DataFrame(confusion.eval({X: X_train, Y: Y_train})), range(2),range(2))
#        sn.set(font_scale=1.4)#for label size
#        sn.heatmap(df, annot=True,annot_kws={"size": 16})
        
        return parameters
    

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
    dic_transco_dic, X_train = input_transco.label_to_num(orig_data, string_labels)
    X_train = X_train.fillna(0)
    X_train = input_transco.normalize(X_train)
    X_train = X_train.fillna(0)
    
    #Getting the ouput for the Training Set
    mth_data = pd.read_csv('sample_svcg_2016.txt', header = None, sep = '|')
    Y_train = data_processor.get_training_output_binary(mth_data)
    Y_train = Y_train.reindex(X_train.index)
    #PB here, so add this line in meantine
    Y_train = Y_train.fillna(0)
    
    #Transpose format, switch to numpy
    X_train = X_train.T.values
    Y_train = Y_train.T.values
    
    Y_train = Y_train.astype(int)
    Y_train = convert_to_one_hot(Y_train, 2)
    
    #parameters = model(X_train, Y_train)
    
    
    #Test Set
    orig_data_test = pd.read_csv('sample_orig_2017.txt', header = None, sep = '|', index_col = 19)
    orig_data_test.columns = orig_col
    
    #Transforming string values to Numerical Values
    string_labels = ['flag_fthb','occpy_sts','channel','ppmt_pnlty','prod_type','st', \
                  'prop_type','loan_purpose','seller_name','servicer_name', 'flag_sc']
    X_test= input_transco.label_to_num_test(orig_data_test, string_labels, dic_transco_dic)
    X_test = X_test.fillna(0)
    X_test = input_transco.normalize(X_test)
    X_test = X_test.fillna(0)
    
    #Getting the ouput for the Training Set
    mth_data_test = pd.read_csv('sample_svcg_2017.txt', header = None, sep = '|')
    Y_test = data_processor.get_test_output_binary(mth_data_test)
    Y_test = Y_test.reindex(X_test.index)
    #PB here, so add this line in meantine
    Y_test = Y_test.fillna(0)

    #Transpose format, switch to numpy
    X_test = X_test.T.values
    Y_test = Y_test.T.values
    
    Y_test = Y_test.astype(int)
    Y_test = convert_to_one_hot(Y_test, 2)
    
    costs = []
    parameters = model(X_train, Y_train, X_test, Y_test, costs)
    
    
    
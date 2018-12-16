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
from sklearn import metrics
import time

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
        
#    W1 = tf.get_variable("W1", [18,n_x], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
#    b1 = tf.get_variable("b1", [18,1], initializer = tf.zeros_initializer())
#    W2 = tf.get_variable("W2", [12,18], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
#    b2 = tf.get_variable("b2", [12,1], initializer = tf.zeros_initializer())
#    W3 = tf.get_variable("W3", [n_y,12], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
#    b3 = tf.get_variable("b3", [n_y,1], initializer = tf.zeros_initializer())
#
#    parameters = {"W1": W1,
#                  "b1": b1,
#                  "W2": W2,
#                  "b2": b2,
#                  "W3": W3,
#                  "b3": b3}

#    W1 = tf.get_variable("W1", [25,n_x], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
#    b1 = tf.get_variable("b1", [25,1], initializer = tf.zeros_initializer())
#    W2 = tf.get_variable("W2", [20,25], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
#    b2 = tf.get_variable("b2", [20,1], initializer = tf.zeros_initializer())
#    W3 = tf.get_variable("W3", [15,20], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
#    b3 = tf.get_variable("b3", [15,1], initializer = tf.zeros_initializer())
#    W4 = tf.get_variable("W4", [10,15], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
#    b4 = tf.get_variable("b4", [10,1], initializer = tf.zeros_initializer())
#    W5 = tf.get_variable("W5", [n_y,10], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
#    b5 = tf.get_variable("b5", [n_y,1], initializer = tf.zeros_initializer())
#
#    parameters = {"W1": W1,
#                  "b1": b1,
#                  "W2": W2,
#                  "b2": b2,
#                  "W3": W3,
#                  "b3": b3,
#                  "W4": W4,
#                  "b4": b4,
#                  "W5": W5,
#                  "b5": b5}

    W1 = tf.get_variable("W1", [40,n_x], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b1 = tf.get_variable("b1", [40,1], initializer = tf.zeros_initializer())
    W2 = tf.get_variable("W2", [30,40], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b2 = tf.get_variable("b2", [30,1], initializer = tf.zeros_initializer())
    W3 = tf.get_variable("W3", [25,30], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b3 = tf.get_variable("b3", [25,1], initializer = tf.zeros_initializer())
    W4 = tf.get_variable("W4", [20,25], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b4 = tf.get_variable("b4", [20,1], initializer = tf.zeros_initializer())
    W5 = tf.get_variable("W5", [15,20], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b5 = tf.get_variable("b5", [15,1], initializer = tf.zeros_initializer())
    W6 = tf.get_variable("W6", [12,15], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b6 = tf.get_variable("b6", [12,1], initializer = tf.zeros_initializer())
    W7 = tf.get_variable("W7", [8,12], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b7 = tf.get_variable("b7", [8,1], initializer = tf.zeros_initializer())
    W8 = tf.get_variable("W8", [3,8], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b8 = tf.get_variable("b8", [3,1], initializer = tf.zeros_initializer())
    W9 = tf.get_variable("W9", [n_y,3], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b9 = tf.get_variable("b9", [n_y,1], initializer = tf.zeros_initializer())
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3,
                  "W4": W4,
                  "b4": b4,
                  "W5": W5,
                  "b5": b5,
                  "W6": W6,
                  "b6": b6,
                  "W7": W7,
                  "b7": b7,
                  "W8": W8,
                  "b8": b8,
                  "W9": W9,
                  "b9": b9}
                  
    return parameters

def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX
    
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"....
                  the shapes are given in initialize_parameters

    Returns:
    Zf -- the output of the last LINEAR unit
    """
    
#    W1 = parameters['W1']
#    b1 = parameters['b1']
#    W2 = parameters['W2']
#    b2 = parameters['b2']
#    W3 = parameters['W3']
#    b3 = parameters['b3']
#    
#    Z1 = tf.add(tf.matmul(W1,X), b1)
#    A1 = tf.nn.relu(Z1)
#    Z2 = tf.add(tf.matmul(W2,A1), b2)
#    A2 = tf.nn.relu(Z2)
#    Zf = tf.add(tf.matmul(W3,A2), b3)


#    W1 = parameters['W1']
#    b1 = parameters['b1']
#    W2 = parameters['W2']
#    b2 = parameters['b2']
#    W3 = parameters['W3']
#    b3 = parameters['b3']
#    W4 = parameters['W4']
#    b4 = parameters['b4']
#    W5 = parameters['W5']
#    b5 = parameters['b5']
#    
#    Z1 = tf.add(tf.matmul(W1,X), b1)
#    A1 = tf.nn.relu(Z1)
#    Z2 = tf.add(tf.matmul(W2,A1), b2)
#    A2 = tf.nn.relu(Z2)
#    Z3 = tf.add(tf.matmul(W3,A2), b3)
#    A3 = tf.nn.relu(Z3)
#    Z4 = tf.add(tf.matmul(W4,A3), b4)
#    A4 = tf.nn.relu(Z4)
#    Zf = tf.add(tf.matmul(W5,A4), b5)


    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    W4 = parameters['W4']
    b4 = parameters['b4']
    W5 = parameters['W5']
    b5 = parameters['b5']
    W6 = parameters['W6']
    b6 = parameters['b6']
    W7 = parameters['W7']
    b7 = parameters['b7']
    W8 = parameters['W8']
    b8 = parameters['b8']
    W9 = parameters['W9']
    b9 = parameters['b9']
    
    keep_prob = 0.8
    Z1 = tf.add(tf.matmul(W1,X), b1)
    A1 = tf.nn.relu(Z1)
    # apply DropOut to hidden layer
    A1 = tf.nn.dropout(A1, keep_prob)
    Z2 = tf.add(tf.matmul(W2,A1), b2)
    A2 = tf.nn.relu(Z2)
    A2 = tf.nn.dropout(A2, keep_prob)
    Z3 = tf.add(tf.matmul(W3,A2), b3)
    A3 = tf.nn.relu(Z3)
    A3 = tf.nn.dropout(A3, keep_prob)
    Z4 = tf.add(tf.matmul(W4,A3), b4)
    A4 = tf.nn.relu(Z4)
    A4 = tf.nn.dropout(A4, keep_prob)
    Z5 = tf.add(tf.matmul(W5,A4), b5)
    A5 = tf.nn.relu(Z5)
    A5 = tf.nn.dropout(A5, keep_prob)
    Z6 = tf.add(tf.matmul(W6,A5), b6)
    A6 = tf.nn.relu(Z6)
    A6 = tf.nn.dropout(A6, keep_prob)
    Z7 = tf.add(tf.matmul(W7,A6), b7)
    A7 = tf.nn.relu(Z7)
    A7 = tf.nn.dropout(A7, keep_prob)
    Z8 = tf.add(tf.matmul(W8,A7), b8)
    A8 = tf.nn.relu(Z8)
    A8 = tf.nn.dropout(A8, keep_prob)
    Zf = tf.add(tf.matmul(W9,A8), b9)
    
    return Zf


def compute_cost(Zf, Y, parameters):
    """
    Computes the cost
    
    Arguments:
    Z3 -- output of forward propagation (output of the last LINEAR unit)
    Y -- "true" labels vector placeholder, same shape as Z3
    
    Returns:
    cost - Tensor of the cost function
    """
    
    logits = tf.transpose(Zf)
    labels = tf.transpose(Y)
    
    #Adding Weights to the positive examples (Defaulting)
    class_weights = tf.constant([[1.0, 3.]])
    weights = tf.reduce_sum(class_weights * labels, axis=1)
    unweighted_cost = tf.nn.softmax_cross_entropy_with_logits_v2(logits = logits, labels = labels)
    weighted_cost = unweighted_cost * weights

    # Loss function using L2 Regularization
    n = int(len(parameters)/2)
    regularizer = 0
    for i in range(1, n+1):
        regularizer += 1/n * tf.nn.l2_loss(parameters['W' + str(i)])
    lambd = 0.005
    
    cost = tf.reduce_mean(weighted_cost+ lambd * regularizer)
    
    return cost


def model(X_train, Y_train, X_test, Y_test, costs, learning_rate = 0.001,#learning_rate = 0.0001,
          num_epochs = 200, minibatch_size = 2048, print_cost = True):
          #num_epochs = 100, minibatch_size = 32, print_cost = True):
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
    
    Zf = forward_propagation(X, parameters)
    
    cost = compute_cost(Zf, Y, parameters)

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
        correct_prediction = tf.equal(tf.argmax(Zf), tf.argmax(Y))
        #correct_prediction = tf.equal(tf.argmax(Z3[1:,]), tf.argmax(Y[1:,]))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
        
        
        confusion = tf.confusion_matrix(labels=tf.argmax(Y), predictions=tf.argmax(Zf), num_classes=2)
        print("Train Confusion Matrix:")
        print(pd.DataFrame(confusion.eval({X: X_train, Y: Y_train}), index = ['Actual Survival', 'Actual Default'], columns = ['Predicted Survival', 'Predicted Default']))
        print("Test Confusion Matrix:")
        print(pd.DataFrame(confusion.eval({X: X_test, Y: Y_test}), index = ['Actual Survival', 'Actual Default'], columns = ['Predicted Survival', 'Predicted Default']))
        
#        auc = tf.metrics.auc(labels=tf.argmax(Y), predictions=tf.argmax(Z3))
#        print ("Train AUC:", auc.eval({X: X_train, Y: Y_train}))
#        print ("Test AUC:", auc.eval({X: X_test, Y: Y_test}))
        
        
#########This part has been added to compute the AUC of the Model
        #Vector of probabilities of surviving
        Y_train_pred = tf.nn.softmax(Zf, axis = 0).eval({X: X_train})[0]
        Y_test_pred = tf.nn.softmax(Zf, axis = 0).eval({X: X_test})[0]
        
        #Vector of status with 0 for Surviving loan and 1 for defaulting loan
        y_train = Y_train[1,]
        y_test = Y_test[1,]
        
        fpr_train, tpr_train, thresholds_train = metrics.roc_curve(y_train, Y_train_pred, pos_label=0)
        print("Train AUC: ", metrics.auc(fpr_train, tpr_train) )

    
        fpr_test, tpr_test, thresholds_test = metrics.roc_curve(y_test, Y_test_pred , pos_label=0)
        print("Test AUC: ", metrics.auc(fpr_test, tpr_test) )
        
        plt.figure()
        plt.plot(fpr_train, tpr_train, color='darkslategray', lw=2, label='ROC curve Train (area = %0.2f)' % metrics.auc(fpr_train, tpr_train))
        plt.plot(fpr_test, tpr_test, color='darkred', lw=2, label='ROC curve Test (area = %0.2f)' % metrics.auc(fpr_test, tpr_test))
        plt.plot([0, 1], [0, 1], color='goldenrod', lw=2, linestyle='--')


        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC')
        plt.legend(loc="lower right")
        plt.show()
        
        df = pd.DataFrame(confusion.eval({X: X_train, Y: Y_train}))
        sn.set(font_scale=1.4)#for label size
        sn.heatmap(df, annot=True,annot_kws={"size": 16})
        
        return parameters
    

if __name__ == "__main__":
    
    t_start = time.time()
    
    orig_col = ['fico','dt_first_pi','flag_fthb','dt_matr','cd_msa',"mi_pct",'cnt_units','occpy_sts',\
              'cltv','dti','orig_upb','ltv','int_rt','channel','ppmt_pnlty','prod_type','st', \
              'prop_type','zipcode','loan_purpose', 'orig_loan_term','cnt_borr','seller_name'\
              ,'servicer_name', 'flag_sc']
    
    #Train set
    #orig_data = pd.read_csv('sample_orig_2016.txt', header = None, sep = '|', index_col = 19)
    year = 2015
    orig_dataprev = input_transco.aggregate(2014)
    orig_data = input_transco.aggregate(year)
    orig_data = orig_dataprev.append(orig_data)
    
    orig_data.columns = orig_col
    
    #Transforming string values to Numerical Values
    string_labels = ['flag_fthb','occpy_sts','channel','ppmt_pnlty','prod_type','st', \
                  'prop_type','loan_purpose','seller_name','servicer_name', 'flag_sc']
    dic_transco_dic, X_train = input_transco.label_to_num(orig_data, string_labels)
    X_train = X_train.fillna(0)
    X_train = input_transco.normalize(X_train)
    X_train = X_train.fillna(0)
    
    #Getting the ouput for the Training Set
    #mth_data = pd.read_csv('sample_svcg_2016.txt', header = None, sep = '|')
    year = 2015
    mth_data_prev = data_processor.aggregate(2014)
    mth_data = data_processor.aggregate(year)
    mth_data = mth_data_prev.append(mth_data)
    
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
    #orig_data_test = pd.read_csv('sample_orig_2017.txt', header = None, sep = '|', index_col = 19)
    year = 2016
    orig_data_test = input_transco.aggregate(year)
    
    orig_data_test.columns = orig_col
    
    #Transforming string values to Numerical Values
    string_labels = ['flag_fthb','occpy_sts','channel','ppmt_pnlty','prod_type','st', \
                  'prop_type','loan_purpose','seller_name','servicer_name', 'flag_sc']
    X_test= input_transco.label_to_num_test(orig_data_test, string_labels, dic_transco_dic)
    X_test = X_test.fillna(0)
    X_test = input_transco.normalize(X_test)
    X_test = X_test.fillna(0)
    
    #Getting the ouput for the Training Set
    #mth_data_test = pd.read_csv('sample_svcg_2017.txt', header = None, sep = '|')
    year = 2016
    mth_data_test = data_processor.aggregate(year)
    
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
    
    t_end = time.time()
    
    print("Done in: ", round((t_end - t_start)/60,2), " min")
    
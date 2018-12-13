"""
Project:DeepRating
Author: Raphael Abbou
Version: python3
"""

import pandas as pd
import numpy as np

def create_trsc_dic(train_set, label):
    """
    Create equivalence for label from String to Numeric, and transform Training Dataset accordingly
                     
    Arguments:
    train_set
    label -- the label of the input we want to transform
    
    Returns: 
    transco_dic -- transco dictionary from train set, for a given label
    """
    transco_dic = {}
    
    str_labels = train_set[label].drop_duplicates().values
    num_lab = 1
    for str_lab in str_labels:
        transco_dic[str_lab] = num_lab
        num_lab += 1
    
    return transco_dic
        
def label_to_num(train_set, string_labels):
    """
    Create equivalence for label from String to Numeric, and transform Training Dataset accordingly
                     
    Arguments:
    train_set
    label -- the label of the input we want to transform
    
    Returns: 
    dic_transco_dic -- transco dictionary from train set,
    train_set -- the train set, with numeric values in label column
    """
    dic_transco_dic = {}
    
    for label in string_labels:
        #Modifying the label column to numeric value
        transco_dic = create_trsc_dic(train_set, label)
        col = train_set[label].values
        trsc_col = [transco_dic[str_lab] for str_lab in col]
        train_set[label] = trsc_col
        
        #for the label, recording in a dic the equivalences between string values to numeric values
        dic_transco_dic[label] =  dict(transco_dic)
        
    return dic_transco_dic, train_set
    
def label_to_num_test(test_set, string_labels, dic_transco_dic):
    """
    Transforming the Test Set accordingly to the equivalences established with the Training set
                     
    Arguments:
    test_set
    label -- the label of the input we want to transform
    dic_transco_dic -- dic of transco_dic from label_to_num function
    
    Returns: 
    test_set -- the train set, with numeric values in label column
    """
    
    for label in string_labels:
        #Modifying the label column to numeric value
        transco_dic = dic_transco_dic[label]
        col_elems = test_set[label].drop_duplicates().values
        
        col = test_set[label].values
        
        # We may have a string label that exist in out train set, but not in our test set
        # Hence we create the equivalence string -> 0 if such a string was not in our train set
        # However, such case should be very rare
        for str_lab in col_elems:
            if str_lab not in transco_dic.keys():
                transco_dic[str_lab] = 0
        
        trsc_col = [transco_dic[str_lab] for str_lab in col]
        test_set[label] = trsc_col
        
        #for the label, recording in a dic the equivalences between string values to numeric values
        dic_transco_dic[label] =  dict(transco_dic)
        
    return test_set 

def na_change(x):
    if (x == 9) or (x == 99) or (x == 999) or (x == 9999):
        return 0
    else:
        return x
    
def managing_NA(X):
    X = X.transform(lambda x: na_change(x))
    return X


def normalize(X):
    return (X-X.mean())/X.std()


if __name__ == "__main__":
    orig_col = ['fico','dt_first_pi','flag_fthb','dt_matr','cd_msa',"mi_pct",'cnt_units','occpy_sts',\
                  'cltv','dti','orig_upb','ltv','int_rt','channel','ppmt_pnlty','prod_type','st', \
                  'prop_type','zipcode','loan_purpose', 'orig_loan_term','cnt_borr','seller_name'\
                  ,'servicer_name', 'flag_sc']
    
    orig_data = pd.read_csv('sample_orig_2016.txt', header = None, sep = '|', index_col = 19)
    orig_data.columns = orig_col

    
    create_trsc_dic(orig_data, 'prop_type')
    
    
    string_labels = ['flag_fthb','occpy_sts','channel','ppmt_pnlty','prod_type','st', \
                  'prop_type','loan_purpose','seller_name','servicer_name', 'flag_sc']
    
    dic_transco_dic, train_set = label_to_num(orig_data, string_labels)
    
    
    #Test Set
    orig_data_test = pd.read_csv('sample_orig_2017.txt', header = None, sep = '|', index_col = 19)
    orig_data_test.columns = orig_col
    
    #Transforming string values to Numerical Values
    string_labels = ['flag_fthb','occpy_sts','channel','ppmt_pnlty','prod_type','st', \
                  'prop_type','loan_purpose','seller_name','servicer_name', 'flag_sc']
    X_test= label_to_num_test(orig_data_test, string_labels, dic_transco_dic)
    X_test = X_test.fillna(0)
    X_test = normalize(X_test)
    
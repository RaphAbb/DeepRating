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
    transco_dic -- transco dictionary from train set
    """
    transco_dic = {}
    
    str_labels = train_set[label].drop_duplicates().values
    num_lab = 1
    for str_lab in str_labels:
        transco_dic[str_lab] = num_lab
        num_lab += 1
    
    return transco_dic
        
def label_to_num(train_set, string_labels):
    
    for label in string_labels:
        transco_dic = create_trsc_dic(train_set, label)
        col = train_set[label].values
        trsc_col = [transco_dic[str_lab] for str_lab in col]
        train_set[label] = trsc_col
    
    return train_set
    
def prop_type_trsc(test_set, train_set, label):
    """
    Get the Property State from String to Numeric for the Test Set
                     
    Arguments:
    test_set
    train_set
    label -- the label of the input we want to transform
    
    Returns: 
    data with numerical value for Property Type
    """
    pass
    
    

if __name__ == "main":
    orig_col = ['fico','dt_first_pi','flag_fthb','dt_matr','cd_msa',"mi_pct",'cnt_units','occpy_sts',\
                  'cltv','dti','orig_upb','ltv','int_rt','channel','ppmt_pnlty','prod_type','st', \
                  'prop_type','zipcode','loan_purpose', 'orig_loan_term','cnt_borr','seller_name'\
                  ,'servicer_name', 'flag_sc']
    
    orig_data = pd.read_csv('sample_orig_2017.txt', header = None, sep = '|', index_col = 19)
    orig_data.columns = orig_col

    
    create_trsc_dic(orig_data, 'prop_type')
    
    
    string_labels = ['flag_fthb','occpy_sts','channel','ppmt_pnlty','prod_type','st', \
                  'prop_type','loan_purpose','seller_name','servicer_name', 'flag_sc']
    
    train_set = label_to_num(orig_data, string_labels)
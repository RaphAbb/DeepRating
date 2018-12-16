"""
Project:DeepRating
Author: Raphael Abbou
Version: python3
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

CUT_OFF_DATE = "201803"

orig_col = ['fico','dt_first_pi','flag_fthb','dt_matr','cd_msa',"mi_pct",'cnt_units','occpy_sts',\
              'cltv','dti','orig_upb','ltv','int_rt','channel','ppmt_pnlty','prod_type','st', \
              'prop_type','zipcode','loan_purpose', 'orig_loan_term','cnt_borr','seller_name'\
              ,'servicer_name', 'flag_sc']

orig_data = pd.read_csv('sample_orig_2016.txt', header = None, sep = '|', index_col = 19)
orig_data.columns = orig_col



#svcg_cols = ['id_loan','svcg_cycle','current_upb','delq_sts','loan_age','mths_remng', 'repch_flag',\
#             'flag_mod', 'cd_zero_bal', 'dt_zero_bal','current_int_rt','non_int_brng_upb',\
#             'dt_lst_pi','mi_recoveries', 'net_sale_proceeds','non_mi_recoveries','expenses','legal_costs',\ 
#             'maint_pres_costs','taxes_ins_costs','misc_costs','actual_loss', 'modcost', 'stepmod_ind', 'dpm_ind']



def get_training_output(mth_data):
    ''' Outputs management
        Arguments:
        mth_data -- raw data frame of monthly data for all loans ID
        
        Returns:
        Y_train, training outputs of shape (C = 7: number of classes, m: number of loans ID)
        
        Notes:
        9th column is the zero balance type
        10th column is the zero balance effective date
    '''
    
    ####Getting Training Ouputs
    #Selecting the data before 2017
    train_set = mth_data[np.around(mth_data[1]/100, decimals=1) <2016]
    #Getting the list of defaulting loans IDs, by looking if the defaulting date is empty or not
    dflt_loans = train_set[train_set[9].notnull()][0]
    #Getting the associated zero balance code, when the defaulting date is not empty
    dflt_code = np.around(train_set[train_set[9].notnull()][8])
    
    #Non defaulting loans IDs
    non_dflt_loans = [loan for loan in train_set[0].drop_duplicates().values if loan not in dflt_loans.values]
    non_dflt_loans = pd.DataFrame(data = [], index = non_dflt_loans)
    
    Y_train = pd.DataFrame(data = dflt_code.values, index = dflt_loans)
    #We add the code 0 for a non-defaulting loan over the time priod considered
    Y_train = Y_train.append(non_dflt_loans).fillna(0)
    Y_train.columns = ["outputs"]
    
    return Y_train



def get_test_output(mth_data):
    ''' Outputs management
        Arguments:
        mth_data -- raw data frame of monthly data for all loans ID
        
        Returns:
        Y_test, training outputs of shape (C = 7: number of classes, m: number of loans ID)
        
        Notes:
        9th column is the zero balance type
        10th column is the zero balance effective date
    '''
    ####Getting Test Ouputs
    #Selecting the data after 2017
    test_set = mth_data[np.around(mth_data[1]/100, decimals=1) >= 2016]
    #Getting the list of defaulting loans IDs
    dflt_loans = test_set[test_set[9].notnull()][0]
    #Getting the associated zero balance code
    dflt_code = np.around(test_set[test_set[9].notnull()][8])
    
    non_dflt_loans = [loan for loan in test_set[0].drop_duplicates().values if loan not in dflt_loans.values]
    non_dflt_loans = pd.DataFrame(data = [], index = non_dflt_loans)
    
    Y_test = pd.DataFrame(data = dflt_code.values, index = dflt_loans)
    #We add the code 0 for a non-defaulting loan over the time priod considered
    Y_test = Y_test.append(non_dflt_loans).fillna(0)
    Y_test.columns = ["outputs"]
    
    return Y_test

def formatting_ouput(Y):
    pass

def aggregate(year):
    data = pd.read_csv("historical_data1_time_Q" + str(1) + str(year) + ".txt", header = None, sep = '|')
    for quarter in range(2,5):
        try:
            data.append(pd.read_csv("historical_data1_time_Q" + str(quarter) + str(year) + ".txt", header = None, sep = '|'))
        except:
            pass
    return data

def switch_to_binary(Y_train):
    return pd.DataFrame(Y_train['outputs'].apply(lambda x: 0 if x == 0 else 1), columns = ['outputs'])

def get_training_output_binary(mth_data):
    Y = get_training_output(mth_data)
    return switch_to_binary(Y)
    
def get_test_output_binary(mth_data):
    Y = get_test_output(mth_data)
    return switch_to_binary(Y)




if __name__ == "__main__":
#    year = 2016
#    #mth_data = pd.read_csv('sample_svcg_2016.txt', header = None, sep = '|')
#    mth_data = aggregate(year)
#    Y_train = get_training_output(mth_data)
#    #print(Y_train.head())
    
    year = 2017
    mth_data = aggregate(year)
    Y_train = get_test_output(mth_data)
    
    hist = Y_train.hist()
    for x in hist[0]:
        x.set_xlabel("Default Types")
        x.set_ylabel("Number of Loans")
        x.set_title("Class repartition for " + str(year))
    
    Y_train = switch_to_binary(Y_train)
    
    
    dft = Y_train[Y_train['outputs'] == 1].count()
    ndft = Y_train[Y_train['outputs'] == 0].count()
    
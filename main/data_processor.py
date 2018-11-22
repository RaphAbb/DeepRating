"""
Author: Raphael Abbou
Version: python3
"""

import pandas as pd
import numpy as np


orig_col = ['fico','dt_first_pi','flag_fthb','dt_matr','cd_msa',"mi_pct",'cnt_units','occpy_sts',\
              'cltv','dti','orig_upb','ltv','int_rt','channel','ppmt_pnlty','prod_type','st', \
              'prop_type','zipcode','loan_purpose', 'orig_loan_term','cnt_borr','seller_name'\
              ,'servicer_name', 'flag_sc']

orig_data = pd.read_csv('sample_orig_2017.txt', header = None, sep = '|', index_col = 19)
orig_data.columns = orig_col



#svcg_cols = ['id_loan','svcg_cycle','current_upb','delq_sts','loan_age','mths_remng', 'repch_flag',\
#             'flag_mod', 'cd_zero_bal', 'dt_zero_bal','current_int_rt','non_int_brng_upb',\
#             'dt_lst_pi','mi_recoveries', 'net_sale_proceeds','non_mi_recoveries','expenses','legal_costs',\ 
#             'maint_pres_costs','taxes_ins_costs','misc_costs','actual_loss', 'modcost', 'stepmod_ind', 'dpm_ind']

''' 9th column is the zero balance type
    10th column is the zero balance effective date
'''
mth_data = pd.read_csv('sample_svcg_2017.txt', header = None, sep = '|')

CUT_OFF_DATE = "201803"

####Getting Training Ouputs
#Selecting the data before 2018
train_set = mth_data[np.around(mth_data[1]/100, decimals=1) != 2018]
#Getting the list of defaulting loans IDs
dflt_loans = train_set[train_set[9].notnull()][0]
#Getting the associated zero balance code
dflt_code = np.around(train_set[train_set[9].notnull()][8])

loans = pd.DataFrame(data = [], index = train_set[0].drop_duplicates().values)

Y_train = pd.DataFrame(data = dflt_code.values, index = dflt_loans)
#We add the code 0 for a non-defaulting loan over the time priod considered
Y_train = Y_train.append(loans).fillna(0)


####Getting Training Ouputs
#Selecting the data after 2018
test_set = mth_data[np.around(mth_data[1]/100, decimals=1) == 2018]
#Getting the list of defaulting loans IDs
dflt_loans = test_set[test_set[9].notnull()][0]
#Getting the associated zero balance code
dflt_code = np.around(test_set[test_set[9].notnull()][8])

loans = pd.DataFrame(data = [], index = test_set[0].drop_duplicates().values)

Y_test = pd.DataFrame(data = dflt_code.values, index = dflt_loans)
#We add the code 0 for a non-defaulting loan over the time priod considered
Y_test = Y_test.append(loans).fillna(0)

 
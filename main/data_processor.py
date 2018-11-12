"""
Author: Raphael Abbou
Version: python3
"""

import pandas as pd
import numpy as np


orig_col = ['fico','dt_first_pi','flag_fthb','dt_matr','cd_msa',"mi_pct",'cnt_units','occpy_sts',\
              'cltv','dti','orig_upb','ltv','int_rt','channel','ppmt_pnlty','prod_type','st', \
              'prop_type','zipcode','id_loan','loan_purpose', 'orig_loan_term','cnt_borr','seller_name'\
              ,'servicer_name', 'flag_sc']

#svcg_cols = ['id_loan','svcg_cycle','current_upb','delq_sts','loan_age','mths_remng', 'repch_flag',\
#             'flag_mod', 'cd_zero_bal', 'dt_zero_bal','current_int_rt','non_int_brng_upb',\
#             'dt_lst_pi','mi_recoveries', 'net_sale_proceeds','non_mi_recoveries','expenses','legal_costs',\ 
#             'maint_pres_costs','taxes_ins_costs','misc_costs','actual_loss', 'modcost', 'stepmod_ind', 'dpm_ind']

data = pd.read_csv('sample_orig_2017.txt', header = None, sep = '|')
data.columns = orig_col
#pd.read_csv('sample_svcg_2017.txt', header = False, sep = '|')
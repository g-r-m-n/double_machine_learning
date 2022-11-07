# %% setup

# set run paramters
n_rep = 3    # number of repetitions.
PRINT = 0    # print (intermediate) results.
theta = 0.5  # true ATE parameter.
n_obs = 1000 # number of observations.
dim_x = 20   # Number of explanatory (confunding) variables.
IV_DGP         = 1 # default: 1. Use a IV data-generating process.
NON_LINEAR_DGP = 1 # default: 1. Use a non-linear data-generating process.
OLS_     = 1 # estimate the OLS model.
TWO_SLS_ = 1 # estimate the 2SLS model.
DML_PLIV_= 1 # estimate the DML-PLIV model.
DML_IIV_ = 1 # estimate the DML-IIV model.


# load libraries
import numpy as np
import pandas as pd
from doubleml.datasets import make_iivm_data, make_pliv_CHS2015
import sys 
pth_to_utils = 'C:/DEV/DML/src/'
# load utility functions
sys.path.append(pth_to_utils+'/utils/')
from utility import *
# reload functions from utility
from importlib import reload
reload(sys.modules['utility'])    

np.random.seed(4444)

# %% model specification

model_index=[]
if OLS_:
    model_index.append('OLS')
if TWO_SLS_:
    model_index.append('2SLS')
if DML_PLIV_:
    model_index.append('DML-PLIV')    
if DML_IIV_:
    model_index.append('DML-IIV')
    
    
# Initialize the result dataset: 
results_rep = pd.DataFrame()




# %% Iterate through repetitions:
for i_rep in range(n_rep): 
    print('\nRepetition '+str(i_rep+1)+' from '+str(n_rep))
    results_rep.loc[i_rep,'rep'] = i_rep+1
    # Generate data
    # linear DGP    
    if not NON_LINEAR_DGP and not IV_DGP:
        data = make_plr_CCDDHNR2018(alpha=theta, n_obs=n_obs, dim_x=dim_x, return_type='DataFrame')
    # non-linear DGP    
    if NON_LINEAR_DGP and not IV_DGP:
        data = make_irm_data(theta=theta, n_obs=n_obs, dim_x=dim_x, return_type='DataFrame')    
    # non-linear IV DGP
    if NON_LINEAR_DGP and IV_DGP:
        data = make_iivm_data(theta=theta, n_obs=n_obs, dim_x=dim_x, alpha_x=1.0, return_type='DataFrame')
    # linear IV DGP    
    if not NON_LINEAR_DGP and IV_DGP:
        data = make_pliv_CHS2015(alpha=theta, n_obs=n_obs, dim_x=dim_x, dim_z=1, return_type='DataFrame')
    # print data descriptions:
    if PRINT:
        print(data.describe())
       
    # interate over the model objects:
    model_object_list = []
    for m in model_index:
        # Initialize the model object:
        model_object_m = model_object(m.lower())
        # update the data for the model objects:
        model_object_m.update_data(data)
        # fit the model objects:
        model_object_m.fit()
        # collect results: 
        model_object_m.collect_results(results_rep, i_rep)
        # print the detailed results if wanted:
        if PRINT:
            print(model_object_m.fitted_model.summary())  
        # save the fitted model object    
        model_object_list.append(model_object_m)
        
            
            
    # print and plot intermediate results:     
    if 1:        
        res_stats = get_res_stats(results_rep, model_index, theta) 
        # plot the ate estimations
        plot_ate_est(results_rep, theta, model_index, max_int_x = n_rep)
            
        
# %% overall results
print(results_rep)


# plot the ate estimations
plot_ate_est(results_rep, theta, model_index)
    

if 1:
    res_stats = get_res_stats(results_rep, model_index, theta)
    




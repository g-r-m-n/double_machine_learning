# %% setup
# pip install numpy pandas doubleml datetime matplotlib

# set run paramters
SAVE_OUTPUT = 0 # default: 1. Save the output of the script.
SCENARIOS   = [1] # default [1, 2, 3, 4]. The list of Scenarios to run.
#IV_DGP         = 0 # default: 1. Use a IV data-generating process.
#NON_LINEAR_DGP = 0 # default: 1. Use a non-linear data-generating process (DGP) and otherwise a parial linear DGP.
ESTIMATE   = 1 # default: 1. Run the estimation process or otherwise re-load results.
n_rep = 30   #default: 1000 number of repetitions.
PRINT = 0    # print (intermediate) results.
theta = 0.5  # true ATE parameter.
n_obs = 1000 # number of observations.
dim_x = 20   # Number of explanatory (confunding) variables.
n_fold= 2    # Number of folds for ML model cross-fitting.
TUNE_MODEL = 1 # default: 1. Tune the model using a n_fold-fold cross-validation with grid search
# models to consider using the first replication.
OLS_     = 1 # estimate the OLS model.
OLS_PO_  = 1 # estimate the OLS partialed-out model.
TWO_SLS_ = 1 # estimate the 2SLS model.
NAIVE_ML_= 1 # estimate the naive ML model.
DML_PLR_ = 1 # estimate the DML-PLR model.
DML_PLIV_= 1 # estimate the DML-PLIV model.
DML_IRM_ = 1 # estimate the DML-IRM model.
DML_IIV_ = 1 # estimate the DML-IIV model.


# load libraries
import numpy as np
import pandas as pd
from doubleml.datasets import make_iivm_data, make_pliv_CHS2015, make_plr_CCDDHNR2018, make_irm_data
import sys, os, json
from datetime import date

#root_dir = '/home/studio-lab-user/'
#root_dir ='/mnt/batch/tasks/shared/LS_root/mounts/clusters/grmnzntt1/code/Users/grmnzntt/'
root_dir = 'C:/DEV/'
pth_to_src = root_dir+'DML/src/'
# data:
today = date.today().strftime('%Y%m%d')
# output folders:
path_to_data  = pth_to_src + 'data/'
output_folder = path_to_data+today+'/'
output_folder_plots  = output_folder+'plots/'
output_folder_tables = output_folder+'tables/'
# create output_folders if they do not exist:
os.makedirs(output_folder,exist_ok=True)
os.makedirs(output_folder_plots,exist_ok=True)
os.makedirs(output_folder_tables,exist_ok=True)
# load utility functions
sys.path.append(pth_to_src+'/utils/')
from utility import *
# reload functions from utility
from importlib import reload
reload(sys.modules['utility'])    

np.random.seed(4444)

# specify the parameter grids for tuning:
if TUNE_MODEL:
    grid_list = [{'n_estimators': [400], 'max_features': [5, 10, 15, 20], 'max_depth': [2, 3, 4, 5], 'min_samples_leaf': [ 2, 4, 6]}] # [{'n_estimators': [100], 'max_features': [5,], 'max_depth': [2, 4], 'min_samples_leaf': [ 2]}] #
    param_grids = dict()
    param_grids['ml_g'] = grid_list
    param_grids['ml_m'] = grid_list
    param_grids['ml_l'] = grid_list
    param_grids['ml_r'] = grid_list
    
# run scenarios:
    
for SCENARIO in SCENARIOS:
    print('\n-----------------------------------------------------------------------')
    print('\nRunning Scenario %s'%SCENARIO)
    if SCENARIO   == 1:
        IV_DGP = 0; NON_LINEAR_DGP = 0 
    elif SCENARIO == 2:
        IV_DGP = 0; NON_LINEAR_DGP = 1 
    elif SCENARIO == 3:
        IV_DGP = 1; NON_LINEAR_DGP = 0 
    elif SCENARIO == 4:
        IV_DGP = 1; NON_LINEAR_DGP = 1
        
    # %% model specification
    
    model_index=[]
    if OLS_:
        model_index.append('OLS')
    if OLS_PO_:
        model_index.append('OLS-partialed-out')   
    if TWO_SLS_ and IV_DGP:
        model_index.append('2SLS')   
    if NAIVE_ML_:
        model_index.append('NAIVE-ML') 
    if DML_PLR_:
        model_index.append('DML-PLR')      
    if DML_PLIV_ and IV_DGP :
        model_index.append('DML-PLIV') 
    if DML_IRM_ and NON_LINEAR_DGP:
        model_index.append('DML-IRM')      
    if DML_IIV_ and IV_DGP and NON_LINEAR_DGP:
        model_index.append('DML-IIV')
        
        
    # Initialize the result dataset: 
    results_rep = pd.DataFrame()
    
    
    # %% Iterate through repetitions:
    if ESTIMATE:        
        for i_rep in range(n_rep): 
            print('\nRepetition '+str(i_rep+1)+' from '+str(n_rep))
            results_rep.loc[i_rep,'rep'] = i_rep+1
            # Generate data
            # linear DGP    
            if not NON_LINEAR_DGP and not IV_DGP:
                data = make_plr_CCDDHNR2018(alpha=theta, n_obs=n_obs, dim_x=dim_x, return_type='DataFrame', a_0 = 1, a_1 = 0.25, s_1 = 1, b_0 = 1, b_1 = 0.25, s_2 = 1) #a_0 = 1.5, a_1 = 1.25, s_1 = .1, b_0 = 1, b_1 = 0.25, s_2 = 1) #
            # non-linear DGP    
            elif NON_LINEAR_DGP and not IV_DGP:
                data = make_irm_data(theta=theta, n_obs=n_obs, dim_x=dim_x, R2_d=0.5, R2_y=0.5, return_type='DataFrame')                  
            # linear IV DGP    
            elif not NON_LINEAR_DGP and IV_DGP:
                data = make_pliv_CHS2015(alpha=theta, n_obs=n_obs, dim_x=dim_x, dim_z=1, return_type='DataFrame')        
            # non-linear IV DGP
            elif NON_LINEAR_DGP and IV_DGP:
                data = make_iivm_data(theta=theta, n_obs=n_obs, dim_x=dim_x, alpha_x=1.0, return_type='DataFrame')
        
                
            # print data descriptions:
            if PRINT:
                print(data.describe())
               
            # interate over the model objects:
            model_object_list = []
            for m in model_index:
                # Initialize the model object:
                model_object_m = model_object(m.lower(), n_fold)
                # update the data for the model objects:
                model_object_m.update_data(data)
                # tune the model dml objects:
                if TUNE_MODEL and model_object_m.type_dml:
                    file_name_tuned_parameters = path_to_data+'tuned_parameters_%s_%s.txt'%(SCENARIO,m)
                    # check if tuned parameters already exist:
                    if os.path.isfile(file_name_tuned_parameters): 
                        # reading the data from the file
                        with open(file_name_tuned_parameters) as f:
                            tuned_params = f.read()
                        tuned_params = json.loads(tuned_params)
                        # adjust the number of folds if needed:
                        for i in tuned_params:
                            for j in tuned_params[i].keys():
                                # adjust if the tuned_paramters are not of length n-fold:
                                if len(tuned_params[i][j][0]) != model_object_m.n_folds:
                                    tuned_params[i][j] =                                 [np.repeat(tuned_params[i][j][0][0],model_object_m.n_folds).tolist()]
                                
                        # set the tuned_params
                        model_object_m.model_obj.params.update(tuned_params)
                    else:
                        #tune the parameters
                        #Note that the parameter are tuned globally, i.e., across folds but are stored per fold, whereas each set of paramters is the same per fold.
                        print('\nTune hyper-parameters ...')
                        model_object_m.tune(param_grids)   
                        tuned_params = model_object_m.model_obj.params
                        # save 
                        with open(file_name_tuned_parameters, "w") as fp:
                            json.dump(tuned_params , fp) 
    
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
                plot_ate_est(results_rep, theta, model_index, max_int_x = n_rep, YLIM= None )
                  
    
    # save results_rep
    if ESTIMATE and SAVE_OUTPUT:
        results_rep.to_csv(output_folder+'results_rep'+'Scenario'+str(SCENARIO)+'.csv', index= False)
        
    # load the results:    
    if not ESTIMATE:
        results_rep = pd.read_csv(output_folder+'results_rep'+'Scenario'+str(SCENARIO)+'.csv')
        
    # %% overall results
    print(results_rep)
    
    # plot the ate estimations
    plot_ate_est(results_rep, theta, model_index,  max_int_x = None, output_folder_plots = output_folder_plots, title1 = 'Scenario'+str(SCENARIO), YLIM= None, SAVE_OUTPUT = SAVE_OUTPUT)

    
    if 1:
        res_stats = get_res_stats(results_rep, model_index, theta)
        # save res_stats
        if SAVE_OUTPUT:
            save_to_tex(res_stats, output_folder_tables+'Scenario'+str(SCENARIO), caption='Scenario '+str(SCENARIO), label ='Scenario'+str(SCENARIO), index=True, longtable=False)  
    
    print('\n-----------------------------------------------------------------------')
    


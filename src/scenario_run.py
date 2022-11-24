# %% setup
# pip install numpy pandas doubleml datetime matplotlib xgboost

# set run paramters
SAVE_OUTPUT = 1 # default: 1. Save the output of the script.
SCENARIOS   = [ 1, 2, 3, 4 ] # default [1, 2, 3, 4]. The list of Scenarios to run.
#IV_DGP         = 0 # default: 1. Use a IV data-generating process.
#NON_LINEAR_DGP = 0 # default: 1. Use a non-linear data-generating process (DGP) and otherwise a parial linear DGP.
ESTIMATE   = 1 # default: 1. Run the estimation process or otherwise re-load results.
n_rep = 100    # default: 100. number of repetitions.
PRINT = 0      # default: 0.print (intermediate) results.
theta = 0.5    # default: 0.5. The true ATE parameter.
n_obs = 10000  # default: 10000 number of observations.
dim_x = 20     # default: 20. Number of explanatory (confunding) varOiables.
n_fold= 5      # default: 5. Number of folds for ML model cross-fitting.
TUNE_MODEL = 1 # default: 1. Tune the model using a n_fold-fold cross-validation with grid search
FORCE_TUING_1 = 0 # default: 1. Force tuning at the first repetition.
# models to consider using the first replication.
MODELS ={
'OLS_'     : 1, # estimate the OLS model.
'OLS_PO_'  : 0, # estimate the OLS partialed-out model.
'TWO_SLS_' : 1, # estimate the 2SLS model.
'NAIVE_ML_': 1, # estimate the naive ML model.
'DML_PLR_' : 1, # estimate the DML-PLR model.
'DML_IRM_' : 1, # estimate the DML-IRM model.
'DML_PLIV_': 1, # estimate the DML-PLIV model.
'DML_IIV_' : 1, # estimate the DML-IIV model.
}
#
alog_type_list = ['Lasso', 'RF','XGBoost','NN'] # default: ['Lasso', 'RF','XGBoost','NN']. list of considered ml algorithms.

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
if 1: #TUNE_MODEL:
    # initialize the parameter grid:
    param_grids = dict() 
    # RF:
    # initialize the RF parameter grid    
    param_grids['RF'] = dict()
    # specific the RF regression parameter grid
    param_grids['RF']['reg']  = [{'n_estimators': [100,400], 'max_features': [ 10,  20], 'max_depth': [5,None], 'min_samples_leaf': [1, 4]}] # [{'n_estimators': [100], 'max_features': [5,], 'max_depth': [2, 4], 'min_samples_leaf': [ 2]}] #
    # specific the RF classification parameter grid
    param_grids['RF']['class'] = param_grids['RF']['reg'] 
    # Lasso:
    param_grids['Lasso'] = dict()
    param_grids['Lasso']['reg']   = [{'alpha':np.arange(0.05, 1, 0.05)}]
    param_grids['Lasso']['class'] = [{'C':np.arange(0.05, 1, 0.05)}]
    # XGBoost:
    param_grids['XGBoost'] = dict()
    param_grids['XGBoost']['reg']  = [{'n_estimators': [100], 'max_depth': [2,5,7], 'learning_rate': [0.01,0.1,0.3]}] 
    param_grids['XGBoost']['class'] = param_grids['XGBoost']['reg'] 
    # NN:
    param_grids['NN'] = dict()
    param_grids['NN']['reg']  = [{ 'hidden_layer_sizes':[(20,),(20,20), (20,20,20), (40,),(40,40), (40,40,40) ]}]
    param_grids['NN']['class'] = param_grids['NN']['reg'] 
    
 
# %% run scenarios:
    
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
    for i in alog_type_list:
        model_index = get_model_index(MODELS, model_index=model_index, alog_type=i, NON_LINEAR_DGP=NON_LINEAR_DGP, IV_DGP=IV_DGP)

    # Initialize the result dataset: 
    results_rep = pd.DataFrame()
    
    
    # %% Iterate through repetitions:
    if ESTIMATE:        
        for i_rep in range(n_rep): 
            print('\nRepetition '+str(i_rep+1)+' from '+str(n_rep))
            results_rep.loc[i_rep,'rep'] = i_rep+1
            # Generate data
            # linear DGP    
            if (not NON_LINEAR_DGP) and (not IV_DGP):
                data = make_plr_CCDDHNR2018(alpha=theta, n_obs=n_obs, dim_x=dim_x, return_type='DataFrame', a_0 = 1, a_1 = 0.25, s_1 = 1, b_0 = 1, b_1 = 0.25, s_2 = 1) 
                                            #a_0 = 1.5, a_1 = 1.25, s_1 = .1, b_0 = 1, b_1 = 0.25, s_2 = 3) #
            # non-linear DGP    
            elif NON_LINEAR_DGP and (not IV_DGP):
                data = make_irm_data(theta=theta, n_obs=n_obs, dim_x=dim_x,  return_type='DataFrame'  , R2_d=0.5, R2_y=0.5  )  
                # data = make_irm_data_ext(theta=theta, n_obs=n_obs, dim_x=dim_x,  return_type='DataFrame'  , R2_d=0.5, R2_y=0.5 , s=1) 
                #data = make_irm_data_ext2(theta=theta, n_obs=n_obs, dim_x=dim_x,  return_type='DataFrame'  , R2_d=0.5, R2_y=0.5 , s=1  )
                #data = make_plr_CCDDHNR2018_nl(alpha=theta, n_obs=n_obs, dim_x=dim_x, return_type='DataFrame', a_0 = 1.5, a_1 = 1.25, s_1 = .1, b_0 = 1, b_1 = 0.25, s_2 = 3)     
            # linear IV DGP    
            elif (not NON_LINEAR_DGP) and IV_DGP:
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
                model_type = m.split(' ')[0].lower()
                algo_type =  m.split(' ')[1] if len(m.split(' '))>1 else ''
                param_grids_reg   = param_grids[algo_type]['reg'] if len(m.split(' '))>1 else ''
                param_grids_class = param_grids[algo_type]['class'] if len(m.split(' '))>1 else ''
                # Initialize the model object:
                model_object_m = model_object(model_type, algo_type, n_fold, param_grids_reg, param_grids_class)
                # update the data for the model objects:
                model_object_m.update_data(data)
                # tune the model dml objects:
                if TUNE_MODEL and model_object_m.type_dml:
                    file_name_tuned_parameters = path_to_data+'tuned_parameters_%s_%s.txt'%(SCENARIO,m)
                    # check if tuned parameters already exist:
                    if (os.path.isfile(file_name_tuned_parameters)) and ((FORCE_TUING_1!=1) or (i_rep>0)): 
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
                        if FORCE_TUING_1 != -1:
                            model_object_m.model_obj.params.update(tuned_params)
                    else:
                        #tune the parameters
                        #Note that the parameter are tuned globally, i.e., across folds but are stored per fold, whereas each set of paramters is the same per fold.
                        print('\nTune hyper-parameters for %s ...'%m)
                        model_object_m.tune()   
                        print('\nCompleted tuning.')
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
            save_to_tex(res_stats, output_folder_tables+'Scenario'+str(SCENARIO), caption='Root mean squared error (RMSE), mean absolute error (MAE) and bias of estimated treatment effect and the true value across the replications for the compared models. The last row indicates which model performs best according to RMSE, MAE or bias.', label ='Scenario'+str(SCENARIO), index=True)  
    
    print('\n-----------------------------------------------------------------------')
    


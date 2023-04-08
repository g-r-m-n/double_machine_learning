# -*- coding: utf-8 -*-

# Usage:
    
# Run the file e.g. in Python with a specific argument setting:   
#   python PATH_TO_YOUR_FOLDER/double_machine_learning/src/scenario_run.py --project_folder PATH_TO_YOUR_FOLDER --config_file test_run_config.yaml
# E.g.:
#   python /home/studio-lab-user/double_machine_learning/src/scenario_run.py --project_folder /home/studio-lab-user --config_file default_config.yaml    

# Run the file e.g. in iPython with a specific argument setting:
#   runfile('PATH_TO_YOUR_FOLDER/double_machine_learning/src/scenario_run.py', args=' --project_folder PATH_TO_YOUR_FOLDER --config_file test_run_config.yaml')
# E.g.:
#   runfile('C:/DEV/double_machine_learning/src/scenario_run.py', args=' --project_folder C:/DEV/ --config_file test_run_config.yaml')    


# %% setup
      
# load libraries
import numpy as np
import pandas as pd
from doubleml.datasets import make_iivm_data, make_pliv_CHS2015, make_plr_CCDDHNR2018, make_irm_data
import sys, os, json, argparse
from datetime import date


# set the path to repository:
parser = argparse.ArgumentParser()
parser.add_argument("--project_folder", default= 'C:/DEV/',             type=str)
parser.add_argument("--config_file",    default= 'test_run_config_v2.yaml', type=str) # default_config.yaml
configs = parser.parse_args()
configs.pth_to_src =  os.path.join(configs.project_folder,'double_machine_learning/src/')
# get the path to the source file repository:   
pth_to_src = configs.pth_to_src

# specify output_folders:
# current date:
today = date.today().strftime('%Y%m%d') # use the current date to save date specific outcomes.
# output folders:
path_to_data  = pth_to_src + 'data/'
output_folder = path_to_data+today+'/'
output_folder_plots  = output_folder+'plots/' # folder for plot outcomes
output_folder_tables = output_folder+'tables/'# folder for tabluar outcomes
# create output_folders if they do not exist:
os.makedirs(output_folder,exist_ok=True)
os.makedirs(output_folder_plots,exist_ok=True)
os.makedirs(output_folder_tables,exist_ok=True)

# load utility functions
sys.path.append(pth_to_src+'/utils/')
from utility import *
# reload functions from utility (to load updates while developing)
from importlib import reload
reload(sys.modules['utility'])    

np.random.seed(4444)


# set the run parameter configurations:  
config_params = set_configs(configs) 

#unpack the dictionnary to variables
locals().update(config_params)



# specify the parameter grids for tuning:
if 1: #TUNE_MODEL:
    # initialize the parameter grid:
    param_grids = dict() 
    # RF:
    # initialize the RF parameter grid    
    param_grids['RF'] = dict()
    # specific the RF regression parameter grid
    param_grids['RF']['reg']  = [{'n_estimators': [100,400], 'max_features': [ 10,  20], 'max_depth': [5,None], 'min_samples_leaf': [1, 4]}] 
    # specific the RF classification parameter grid
    param_grids['RF']['class'] = param_grids['RF']['reg'] 
    # Lasso:
    param_grids['Lasso'] = dict()
    param_grids['Lasso']['reg']   = [{'alpha':np.arange(0.01, 1, 0.01)}]
    param_grids['Lasso']['class'] = [{'C':np.arange(0.01, 1, 0.01)}]
    # XGBoost:
    param_grids['XGBoost'] = dict()
    param_grids['XGBoost']['reg']  = [{'n_estimators': [100], 'max_depth': [2,5,7], 'learning_rate': [0.01,0.1,0.3]}] 
    param_grids['XGBoost']['class'] = param_grids['XGBoost']['reg'] 
    # NN:
    param_grids['NN'] = dict()
    param_grids['NN']['reg']  = [{ 'hidden_layer_sizes':[(20,),(10,10),(7,7,7), (40,),(20,20),(14,14,14)]}]
    param_grids['NN']['class'] = param_grids['NN']['reg'] 
    # Dols:
    param_grids['Dols'] = dict()
    param_grids['Dols']['reg']   = [{'fit_intercept':[True]}]
    param_grids['Dols']['class'] = [{'C':[0.0001]}]
 
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
    #  estimation of theta:    
    results_rep      = pd.DataFrame()
    #  prediction of y:
    results_rep_pred = pd.DataFrame()
    
    # %% Iterate through repetitions:
    if ESTIMATE:        
        for i_rep in range(n_rep): 
            print('\n-----------------------------------------------')
            print('\nRepetition '+str(i_rep+1)+' from '+str(n_rep))
            print('\n-----------------------------------------------')
            results_rep.loc[i_rep,'rep'] = i_rep+1
            results_rep_pred.loc[i_rep,'rep'] = i_rep+1
            # Generate data
            # linear DGP    
            if (not NON_LINEAR_DGP) and (not IV_DGP):
                data = make_plr_CCDDHNR2018_II(alpha=theta, n_obs=n_obs, dim_x=dim_x, return_type='DataFrame', add_additional_nonlinearity = add_additional_nonlinearity, a_0 = 1, a_1 = 0.25, s_1 = 1, b_0 = 1, b_1 = 0.25, s_2 = 1) 
            # non-linear DGP    
            elif NON_LINEAR_DGP and (not IV_DGP):
                data = make_irm_data_II(theta=theta, n_obs=n_obs, dim_x=dim_x,  return_type='DataFrame'  , R2_d=0.5, R2_y=0.5, add_additional_nonlinearity = add_additional_nonlinearity )     
            # linear IV DGP    
            elif (not NON_LINEAR_DGP) and IV_DGP:
                data = make_pliv_CHS2015_II(alpha=theta, n_obs=n_obs, dim_x=dim_x, dim_z=1, return_type='DataFrame', add_additional_nonlinearity = add_additional_nonlinearity )        
            # non-linear IV DGP
            elif NON_LINEAR_DGP and IV_DGP:
                data = make_iivm_data_II(theta=theta, n_obs=n_obs, dim_x=dim_x, alpha_x=1.0, return_type='DataFrame', add_additional_nonlinearity = add_additional_nonlinearity )
            
            
            # print data descriptions:
            if PRINT:
                print(data.describe())
            
            # Add constant    
            if 0:
                data['X0'] = 1
                
            # interate over the model objects:
            model_object_list = []
            for m in model_index:
                model_type = m.split(' ')[0].lower()
                algo_type =  m.split(' ')[1] if len(m.split(' '))>1 else ''
                param_grids_reg   = param_grids[algo_type]['reg'] if len(m.split(' '))>1 else ''
                param_grids_class = param_grids[algo_type]['class'] if len(m.split(' '))>1 else ''
                # Initialize the model object:
                model_object_m = model_object(model_type, algo_type, n_fold, param_grids_reg, param_grids_class, score)
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
                                tuned_params_ij =     tuned_params[i][j][0]
                                if (tuned_params_ij is not None) and (len(tuned_params_ij) != model_object_m.n_folds): 
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
                # estimation of theta:    
                model_object_m.collect_results(results_rep, i_rep)
                # prediction of y:
                model_object_m.collect_results_pred(results_rep_pred, i_rep, data)
            
                
            # print and plot intermediate results for y_pred::
            if 1:    
                 res_stats_pred = get_res_stats_agg(results_rep_pred, model_index)     
                    
            # print and plot intermediate results for theta:     
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
    


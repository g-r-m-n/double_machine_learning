from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
#from linearmodels import IV2SLS, IVGMM, OLS
from statsmodels.sandbox.regression.gmm import IV2SLS, OLS
import doubleml as dml
import matplotlib.pyplot as plt
import sklearn.metrics as metrics 
from sklearn.base import clone
import numpy as np
import os
import pandas as pd
from scipy.linalg import toeplitz
_array_alias = ['array', 'np.ndarray', 'np.array', np.ndarray]
_data_frame_alias = ['DataFrame', 'pd.DataFrame', pd.DataFrame]
_dml_data_alias = ['DoubleMLData', dml.DoubleMLData]
#from xgboost import XGBClassifier,XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.linear_model import Lasso, LogisticRegression , LinearRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor 
from sklearn.preprocessing import StandardScaler

import argparse, yaml

def get_configs(default):
    '''Parse the configuration file name from command line or set to default.'''
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", default=default, type=str, help='configuration file for setup specifications')

    configs = parser.parse_args()
    return configs

def set_configs(default='default_config.yaml'):
    '''Set the configuration parameters based on the configuration file inputs'''
    configs = get_configs(default)
    f = open(os.path.join('config',configs.config_file),'rb')
    # load the config dictionnary form the yaml config file:
    config_params = yaml.load(f, Loader=yaml.FullLoader)
    return config_params
    

def get_model_index(MODELS, model_index=[], alog_type='RF', NON_LINEAR_DGP=0, IV_DGP=0):
    
    if MODELS['OLS_'] and ('OLS' not in model_index):
        model_index.append('OLS')
    if MODELS['OLS_PO_'] and ('OLS-partialed-out' not in model_index):
        model_index.append('OLS-partialed-out')   
    if MODELS['TWO_SLS_'] and IV_DGP and ('2SLS' not in model_index):
        model_index.append('2SLS')   
    if MODELS['NAIVE_ML_']:
        model_index.append('NAIVE-ML '+alog_type) 
    if MODELS['DML_PLR_']:
        model_index.append('DML-PLR '+alog_type)      
    if MODELS['DML_IRM_'] and NON_LINEAR_DGP:
        model_index.append('DML-IRM '+alog_type)   
    if MODELS['DML_PLIV_'] and IV_DGP :
        model_index.append('DML-PLIV '+alog_type)         
    if MODELS['DML_IIV_'] and IV_DGP and NON_LINEAR_DGP:
        model_index.append('DML-IIV '+alog_type)
    return(model_index)    



class model_object:
    
    def __init__(self, model_type, algo_type='RF', n_fold = 2, param_grids_reg=[],param_grids_class=[], score = 'partialling out'):
        self.type = model_type
        self.algo_type = algo_type
        # is dml model type?
        self.type_dml  =  self.type in ['dml-plr','dml-pliv','dml-irm','dml-iiv','naive-ml']
        # is linear regression model type?
        self.type_lreg =  self.type in ['2sls','ols','ols-partialed-out']
        self.n_folds = n_fold
        self.param_grids = dict()
        self.param_grids_reg = param_grids_reg
        self.param_grids_class = param_grids_class
        self.score = score
        
    def update_data(self, data) :  
        
        data = data.copy()
        if 0:# scale variables
            input_vars =  [i for i in data.columns if not i.lower().startswith('y')] #data.columns # 
            scaler = StandardScaler()
            scaler_model = scaler.fit(data.loc[:,input_vars])
            data.loc[:,input_vars] = pd.DataFrame(scaler_model.transform(data.loc[:,input_vars]),columns=input_vars)
            
        if self.type_dml:
            # specify the ML models:
            if self.algo_type == 'RF':
                learner_reg = RandomForestRegressor(n_estimators=100, max_features=20, max_depth=5, min_samples_leaf=2)
                
                learner_class = RandomForestClassifier(n_estimators=100, max_features=20, max_depth=5, min_samples_leaf=2)
                
            elif self.algo_type == 'XGBoost':   
                #learner_reg   = XGBRegressor(n_estimators=100, max_depth=5,  learning_rate =0.1)   
                #learner_class = XGBClassifier(n_estimators=100, max_depth=5,  learning_rate =0.1, eval_metric ='logloss', use_label_encoder=False) 
                learner_reg   = LGBMRegressor(n_estimators=100, max_depth=5,  learning_rate =0.1)   
                learner_class = LGBMClassifier(n_estimators=100, max_depth=5,  learning_rate =0.1) 


            elif self.algo_type == 'Lasso':   
                learner_reg   = Lasso(alpha=0.1)   
                learner_class = LogisticRegression(C=0.1, penalty='l1', solver ='saga')
                
            elif self.algo_type == 'Dols':   
                    learner_reg   = LinearRegression() 
                    learner_class = LogisticRegression(C=0.1, penalty='l1', solver ='saga')
                
            elif self.algo_type == 'NN':   
                learner_reg   = MLPRegressor(hidden_layer_sizes=(10,10),max_iter=10000, activation = 'identity')
                learner_class = MLPClassifier(hidden_layer_sizes=(10,10),max_iter=10000)    
                
            iv_vars =[i for i in data.columns if i.lower().startswith('z')]    
            
            
            if self.type == 'dml-plr':
                
                obj_dml_data = dml.DoubleMLData(data, 'y', 'd')
                
                #ml learner for the nuisance function l0(X)=E[Y|X]:
                self.ml_l = clone(learner_reg)
                self.param_grids['ml_l'] = self.param_grids_reg
                
                #ml learner for the nuisance function g0(X)=E[Y-Dθ0|X]:
                if self.score ==   'IV-type': 
                    self.ml_g = clone(learner_reg)
                    self.param_grids['ml_g'] = self.param_grids_reg                
                else:
                    self.ml_g = None
                    
                #ml learner for the nuisance function m0(X)=E[D|X]:
                if len(np.unique(data['d'])) <= 10:  
                    self.ml_m = clone(learner_class)
                    self.param_grids['ml_m'] = self.param_grids_class
                else:                    
                    self.ml_m = clone(learner_reg)
                    self.param_grids['ml_m'] = self.param_grids_reg
                
                self.model_obj  = dml.DoubleMLPLR(obj_dml_data, self.ml_l, self.ml_m, self.ml_g, n_folds = self.n_folds, score = self.score)

            elif self.type == 'dml-irm':
                
                obj_dml_data = dml.DoubleMLData(data, 'y', 'd')
                
                #ml learner for the nuisance function g0(X)=E[Y-Dθ0|X]:
                self.ml_g = clone(learner_reg)
                self.param_grids['ml_g'] = self.param_grids_reg
                
                #ml learner for the nuisance function m0(X)=E[D|X]:
                if len(np.unique(data['d'])) <= 10:  
                    self.ml_m = clone(learner_class)
                    self.param_grids['ml_m'] = self.param_grids_class
                else:                    
                    self.ml_m = clone(learner_reg)
                    self.param_grids['ml_m'] = self.param_grids_reg
                
                self.model_obj = dml.DoubleMLIRM(obj_dml_data, self.ml_g, self.ml_m, n_folds = self.n_folds)
                
            elif self.type == 'dml-pliv':
                
                obj_dml_data = dml.DoubleMLData(data, 'y', 'd', z_cols=iv_vars)
                
                #ml learner for the nuisance function l0(X)=E[Y|X]:
                self.ml_l = clone(learner_reg)
                self.param_grids['ml_l'] = self.param_grids_reg
                
                #ml learner for the nuisance function m0(X)=E[Z|X]:
                self.ml_m = clone(learner_reg)
                self.param_grids['ml_m'] = self.param_grids_reg
                
                #ml learner for the nuisance function r0(X)=E[D|X]:
                self.ml_r = clone(learner_reg)
                self.param_grids['ml_r'] = self.param_grids_reg
                
                #ml learner for the nuisance function g0(X)=E[Y-Dθ0|X]:
                if self.score == 'IV-type': 
                    self.ml_g = clone(learner_reg)
                    self.param_grids['ml_g'] = self.param_grids_reg                   
                else:
                    self.ml_g = None
                    
                self.model_obj = dml.DoubleMLPLIV(obj_dml_data, self.ml_l, self.ml_m, self.ml_r, self.ml_g, n_folds = self.n_folds, score = self.score)
                
            elif self.type == 'dml-iiv':

                obj_dml_data = dml.DoubleMLData(data, 'y', 'd', z_cols=iv_vars)
                
                #ml learner for the nuisance function g0(X)=E[Y-Dθ0|X]:
                self.ml_g = clone(learner_reg)
                self.param_grids['ml_g'] = self.param_grids_reg
                
                #ml learner for the nuisance function m0(X)=E[Z|X]:
                self.ml_m = clone(learner_class)
                self.param_grids['ml_m'] = self.param_grids_class
                
                #ml learner for the nuisance function r0(X)=E[D|X]:
                self.ml_r = clone(learner_class)
                self.param_grids['ml_r'] = self.param_grids_class
                
                self.model_obj = dml.DoubleMLIIVM(obj_dml_data, self.ml_g, self.ml_m, self.ml_r, n_folds = self.n_folds)    
                
            elif self.type =='naive-ml':
                
                obj_dml_data = dml.DoubleMLData(data, 'y', 'd')
                
                #ml learner for the nuisance function l0(X)=E[Y|X]:
                self.ml_l = clone(learner_reg)
                self.param_grids['ml_l'] = self.param_grids_reg
                    
                #ml learner for the nuisance function m0(X)=E[Z|X] or  m0(X)=E[D|X]:
                if len(np.unique(data['d'])) <= 10:  
                    self.ml_m = clone(learner_class)
                    self.param_grids['ml_m'] = self.param_grids_class
                else:                    
                    self.ml_m = clone(learner_reg)
                    self.param_grids['ml_m'] = self.param_grids_reg
                    
                #ml learner for the nuisance function g0(X)=E[Y-Dθ0|X]:
                self.ml_g = clone(learner_reg)   
                self.param_grids['ml_g'] = self.param_grids_reg
                
                self.model_obj = dml.DoubleMLPLR(obj_dml_data, self.ml_l, self.ml_m, self.ml_g, n_folds = self.n_folds, score=non_orth_score_w_g)
                
            else:
                raise ValueError("model type not found.")
            
        if self.type =='ols':
            data.loc[:,'X0'] = 1
            exog = [i for i in data.columns if i.lower().startswith('x')] + ['X0'] 
            exog = pd.unique(exog).tolist()
            
            self.model_obj = OLS(endog=data.loc[:,'y'], exog = data.loc[:,['d']+exog])            
         
        if self.type =='ols-partialed-out':
            
            data.loc[:,'X0'] = 1
            exog = [i for i in data.columns if i.lower().startswith('x')] + ['X0'] 
            exog = pd.unique(exog).tolist()
            res_D_on_X = OLS(endog=data.loc[:,'d'], exog = data.loc[:,exog]).fit().resid 
            data.loc[:,'res_D_on_X'] = res_D_on_X
            
            res_Y_on_X = OLS(endog=data.loc[:,'y'], exog = data.loc[:,exog]).fit().resid 
            
            self.model_obj = OLS(endog = res_Y_on_X, exog = data.loc[:, ['res_D_on_X','X0']] )  
          
        if self.type =='2sls':

            exog = [i for i in data.columns if i.lower().startswith('x')] 
            iv_vars =[i for i in data.columns if i.lower().startswith('z')]
            
            if 1 and len(iv_vars)==0:
                iv_vars = [i+'_sq' for i in exog]
                data[iv_vars] = data[exog]**2
                
            data.loc[:,'X0'] = 1
            exog += ['X0']  
            exog = pd.unique(exog).tolist()
            self.model_obj = IV2SLS(endog=data.loc[:,'y'], exog = data.loc[:,['d']+exog], instrument= data.loc[:,exog+iv_vars])
    
    def tune(self, **kwargs):  
        num_of_parameters_to_tune = len([l[0] for l in self.param_grids.values() if l])
        if self.type_dml and (num_of_parameters_to_tune>0):
           self.model_obj.tune(self.param_grids, n_folds_tune = self.n_folds, **kwargs)
         
       
    def fit(self, **kwargs):
        if self.type_dml:
            self.fitted_model = self.model_obj.fit(store_predictions = True, **kwargs)
            self.fitted_model.interval = self.model_obj.confint()
        if self.type_lreg :
            self.fitted_model = self.model_obj.fit(**kwargs)
            self.fitted_model.interval = self.fitted_model.conf_int()   
            self.fitted_model.coef = self.fitted_model.params
    
    
    def predict(self, **kwargs):
        if self.type_dml :
            y_pred = self.model_obj.predictions
            y_pred = pd.DataFrame({i:y_pred[i].flatten() for i in y_pred.keys()})
            # if 0 and self.model_type == 'dml-irm':
            #     # estimation of d
            #     print('\n'+model_type +':'+algo_type)
            #     print(pd.crosstab(data.loc[:,'d'],(y_pred.ml_m>=0.5)*1))
            #     # estimation of y
            #     y_est = y_pred.ml_g1*y_pred.ml_m + y_pred.ml_g0*(1-y_pred.ml_m)
            #     print(error_stats(data.loc[:,'y'], y_est))
            return y_pred
        
        if self.type_lreg :
            y_pred = self.fitted_model.predict(**kwargs) 
            return y_pred
           
            
    def collect_results(self,results_rep,i_rep): 
        results_rep.loc[i_rep,'coef_'+self.type+' '+self.algo_type] = self.fitted_model.coef[0] 
        results_rep.loc[i_rep,'lower_bound_'+self.type+' '+self.algo_type] = self.fitted_model.interval.iloc[0,0] 
        results_rep.loc[i_rep,'upper_bound_'+self.type+' '+self.algo_type] = self.fitted_model.interval.iloc[0,1] 
        return  results_rep
    
    def collect_results_pred(self,results_rep_pred, i_rep, data): 
        if self.type_dml :
            y_pred = self.predict()
            if self.type == 'dml-plr':
                theta_est = self.fitted_model.coef[0]
                # l0(X) = E[Y|X]
                # m0(X) = E[D|X]
                # g0(X) = E[Y-Dθ0|X]
                #y_pred_1 = theta_est*(data.d - y_pred['ml_m']) + y_pred['ml_l'] 
                if self.score == 'IV-type': 
                    y_pred_1 = theta_est * data.d + y_pred['ml_g']
                else:
                    y_pred_1 =  y_pred['ml_l'] 
                
            elif self.type == 'dml-irm':
                theta_est = self.fitted_model.coef[0]
                # g0(X) = E[Y-Dθ0|X]
                # m0(X) = E[D|X]
                #y_pred_1 = theta_est * data.d + y_pred['ml_g'] 
                y_pred_1 = y_pred['ml_g1']*y_pred['ml_m'] + y_pred['ml_g0']*(1-y_pred['ml_m'])

            elif self.type == 'dml-pliv':
                theta_est = self.fitted_model.coef[0]
                # l0(X) = E[Y|X]
                # m0(X) = E[Z|X]
                # r0(X) = E[D|X]
                # g0(X) = E[Y-Dθ0|X]
                #y_pred_1 = theta_est*(data.d - y_pred['ml_r']) + y_pred['ml_l'] 
                if self.score == 'IV-type': 
                    y_pred_1 = theta_est * data.d + y_pred['ml_g'] 
                else:
                    y_pred_1 =  y_pred['ml_l'] 
                
            elif self.type == 'dml-iiv':
                theta_est = self.fitted_model.coef[0]
                # g0(X) = E[Y-Dθ0|X]
                # m0(X) = E[Z|X]
                # r0(X) = E[D|X]
                #y_pred_1 = theta_est * data.d + y_pred['ml_g'] 
                y_pred_1 = y_pred['ml_g1']*y_pred['ml_m'] + y_pred['ml_g0']*(1-y_pred['ml_m'])
                 
            elif self.type =='naive-ml':
                theta_est = self.fitted_model.coef[0]
                # l0(X) = E[Y|X]
                # m0(X) = E[Z|X] or m0(X)=E[D|X]
                # g0(X) = E[Y-Dθ0|X]
                y_pred_1 = theta_est * data.d+ y_pred['ml_g'] 
                
            else:
                y_pred_1 = None
        if self.type_lreg :
            y_pred_1 = self.predict()
        es1 = error_stats(data.y, y_pred_1)
        for i in es1.keys():
            results_rep_pred.loc[i_rep, i +' '+ self.type+' '+self.algo_type] = es1[i]

        return  results_rep_pred


    


def plot_ate_est(results_rep, theta, model_index, max_int_x = None, output_folder_plots = '', title1 = '', YLIM=[0,1], SAVE_OUTPUT = 0):
    #plt.style.use('ggplot')
    import matplotlib.pyplot as plt
    plt.rcParams['figure.figsize'] = (16.0, 16.0)
    plt.dpi = 100
    nn = len(results_rep)+1
    x = range(1,nn)
    if max_int_x is None:
        max_int_x = nn
    if len(model_index)<2:
        return      
    ncols = 1
    nrows = len(model_index)
    if nrows > 8:
        nrows = 8
        ncols = int(np.ceil(len(model_index)/nrows))
        if nrows < 16:
            ncols = 2 
            nrows = int(np.ceil(len(model_index)/ncols))
        
    fig, axes = plt.subplots(nrows=nrows, ncols= ncols, sharex=True, squeeze=False) 
    m_i = 0
    for j in range(ncols):
        for i in range(nrows):    
            if m_i >= len(model_index):
                break
            else:
                m = model_index[m_i]; m_i += 1
            model_type = m.split(' ')[0].lower()
            algo_type =  m.split(' ')[1] if len(m.split(' '))>1 else ''
            y = results_rep['coef_'+model_type+' '+algo_type]
            asymmetric_error = [abs(results_rep['lower_bound_'+model_type+' '+algo_type].values-y), abs(y-results_rep['upper_bound_'+model_type+' '+algo_type].values)]
            
            
            axes[i,j].errorbar(x, y, yerr=asymmetric_error, fmt='o')
            axes[i,j].set_title(m.upper(), fontsize=16)
            axes[i,j].hlines(theta, 1, max_int_x-1, color='red')
            if YLIM is not None:
                axes[i,j].set_ylim([0.0, 1])
          
            #text size:
            labels = axes[i,j].get_xticklabels() + axes[i,j].get_yticklabels()
            for label in labels:
                #label.set_fontweight('bold')
                label.set_size('16')
    
        
    # Saving plot to pdf file
    if SAVE_OUTPUT:
        plt.savefig(output_folder_plots  +title1+'.pdf', dpi=plt.dpi,bbox_inches="tight")
        #plt.title(title1, fontsize=20)
        plt.savefig(output_folder_plots  +title1+ '.png', dpi=plt.dpi,bbox_inches="tight")


    plt.show()
    
    
def save_to_tex(df, output_folder_file_path, column_format='', caption='', label='', index=False):
    """
    Save a data table to LaTeX in the local folder.
    :param df: (dataframe) dataframe to be saved.
    :param output_folder_file_path: (string) path and file name in the local folder the dataframe is saved to.
    :param column_format: (string) the column format in LaTeX style of the table to be saved. If not provided, the column_format will be automatically generated
    :param caption: (string) the caption to be shown in LaTeX of the table to be saved.
    :param label: (string) the label to be shown in LaTeX of the table to be saved.
    :param index: (boolean) indicating whether the index of the dataframe should be saved.
    :return True: (boolean)  .
    """
    df= df.copy()
    # convert to pandas dataframe if df is not:
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    # create output folder if it does not exit:
    output_folder_path,_ = os.path.split(output_folder_file_path)
    os.makedirs(output_folder_path, exist_ok=True)
    # format column names
    df.columns = df.columns.str.replace("_", " ")
    df.columns = df.columns.str.replace("\n", "")
    if 0:
        df.columns = df.columns.str.capitalize()
    # determine the column format for the output Latex table
    if len(column_format)==0:
        column_format = ['L{'+str(len(df.columns[0])/1.5)+'cm}']+[ 'R{'+str(len(i)/1.5)+'cm}' for i in df.columns[1:] ]
    if index:
        if len(df.index.names)>1:
            df_index_names = df.index.names 
        else:
            df_index_names  = ['Index'] 
        column_format = [ 'L{'+str(len(i)/1.5)+'cm}' for i in  df_index_names ] +column_format
    else:
        df = df.reset_index(drop=True)
    df = df.fillna('')   
    pd.options.styler.format.precision =4
    # transform to string:
    column_format = ''.join(column_format)
    with open(output_folder_file_path+ '.tex', 'w') as texf:
        texf.write(df.style.to_latex(column_format=column_format, label=label,caption=caption, position='H',hrules= True, position_float= "centering"))
    return True
    
   
def error_stats(theta, y_pred):
    if not hasattr(theta, '__len__') or len(theta) ==1:
        y_true = np.repeat(theta, len(y_pred))
    else:
        y_true = theta
    mean_absolute_error=metrics.mean_absolute_error(y_true, y_pred) 
    mse=metrics.mean_squared_error(y_true, y_pred) 
    #median_absolute_error=metrics.median_absolute_error(y_true, y_pred)
    #return {'MSE': round(mse,4),'RMSE': round(np.sqrt(mse),4),'MAE': round(mean_absolute_error,4), 'MEDIAN-AE':round(median_absolute_error,4) } 
    return {'RMSE': round(np.sqrt(mse),4),'MAE': round(mean_absolute_error,4), 'Bias':round(np.mean(y_pred-y_true),4) }    	


def get_res_stats(results_rep, model_index, theta, PRINT = 1, prefix = 'coef_'):
    if 0:#len(results_rep)<= 1:
        return []
    res_stats = pd.DataFrame()
    for m in model_index:
        model_type = m.split(' ')[0].lower()
        algo_type =  m.split(' ')[1] if len(m.split(' '))>1 else ''
        res_stats = pd.concat( [res_stats, pd.DataFrame([error_stats(theta, results_rep[prefix + model_type + ' '+algo_type])],index=[m])], axis=0 )
        
    # add best performing model:
    res_stats2 = res_stats.copy()
    if 'Bias' in res_stats2.columns:
   	    res_stats2['Bias'] = abs(res_stats2['Bias'])
    res_stats.loc['Best',:] =  res_stats2.idxmin()
    if PRINT:
        print('\ntheta: error statistics for the prediction of theta across repetitions:')
        print(res_stats)
    return res_stats


def get_res_stats_agg(results_rep_pred, model_index, PRINT = 1):
    res_stats  = pd.DataFrame()
    mean_stats = results_rep_pred.mean().round(3)
    for m in model_index:
        model_type = m.split(' ')[0].lower()
        algo_type =  m.split(' ')[1] if len(m.split(' '))>1 else ''
        m_ix = mean_stats.index[mean_stats.index.str.endswith(model_type + ' '+algo_type)]
        res_stats_m = pd.DataFrame( mean_stats[m_ix].values, index= [i.split(' ')[0] for i in m_ix], columns=[m]).T
        res_stats = pd.concat( [res_stats, res_stats_m], axis=0 )
        
    # add best performing model:
    res_stats2 = res_stats.copy()
    if 'Bias' in res_stats2.columns:
   	    res_stats2['Bias'] = abs(res_stats2['Bias'])
    res_stats.loc['Best',:] =  res_stats2.idxmin()
    if PRINT:
        print('\ny: average error statistics for the prediction of y across repetitions:')
        print(res_stats)
    return res_stats


def non_orth_score_w_g(y, d, l_hat, m_hat, g_hat, smpls, *kwargs):
    """ author: DoubleML python package.
    Non-orthotogonal score for partial linear model, i.e., the naive partial linear model. """
    u_hat = y - g_hat
    psi_a = -np.multiply(d, d)
    psi_b = np.multiply(d, u_hat)
    return psi_a, psi_b



######################################################################################
# Add modified versions of the used data generating functions from DoubleML package, so that the data generating process can be further altered, if wanted. (Per default it is not.)
######################################################################################

from doubleml.double_ml_data import DoubleMLData


def make_plr_CCDDHNR2018_II(n_obs=500, dim_x=20, alpha=0.5, return_type='DoubleMLData', add_additional_nonlinearity = False, **kwargs):
    """Modified version of make_plr_CCDDHNR2018 from DoubleML package.
    
    Generates data from a partially linear regression model used in Chernozhukov et al. (2018) for Figure 1.
    The data generating process is defined as

    .. math::

        d_i &= m_0(x_i) + s_1 v_i, & &v_i \\sim \\mathcal{N}(0,1),

        y_i &= \\alpha d_i + g_0(x_i) + s_2 \\zeta_i, & &\\zeta_i \\sim \\mathcal{N}(0,1),


    with covariates :math:`x_i \\sim \\mathcal{N}(0, \\Sigma)`, where  :math:`\\Sigma` is a matrix with entries
    :math:`\\Sigma_{kj} = 0.7^{|j-k|}`.
    The nuisance functions are given by

    .. math::

        m_0(x_i) &= a_0 x_{i,1} + a_1 \\frac{\\exp(x_{i,3})}{1+\\exp(x_{i,3})},

        g_0(x_i) &= b_0 \\frac{\\exp(x_{i,1})}{1+\\exp(x_{i,1})} + b_1 x_{i,3}.

    Parameters
    ----------
    n_obs :
        The number of observations to simulate.
    dim_x :
        The number of covariates.
    alpha :
        The value of the causal parameter.
    return_type :
        If ``'DoubleMLData'`` or ``DoubleMLData``, returns a ``DoubleMLData`` object.

        If ``'DataFrame'``, ``'pd.DataFrame'`` or ``pd.DataFrame``, returns a ``pd.DataFrame``.

        If ``'array'``, ``'np.ndarray'``, ``'np.array'`` or ``np.ndarray``, returns ``np.ndarray``'s ``(x, y, d)``.
    **kwargs
        Additional keyword arguments to set non-default values for the parameters
        :math:`a_0=1`, :math:`a_1=0.25`, :math:`s_1=1`, :math:`b_0=1`, :math:`b_1=0.25` or :math:`s_2=1`.

    References
    ----------
    Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W. and Robins, J. (2018),
    Double/debiased machine learning for treatment and structural parameters. The Econometrics Journal, 21: C1-C68.
    doi:`10.1111/ectj.12097 <https://doi.org/10.1111/ectj.12097>`_.
    """
    a_0 = kwargs.get('a_0', 1.)
    a_1 = kwargs.get('a_1', 0.25)
    s_1 = kwargs.get('s_1', 1.)

    b_0 = kwargs.get('b_0', 1.)
    b_1 = kwargs.get('b_1', 0.25)
    s_2 = kwargs.get('s_2', 1.)

    cov_mat = toeplitz([np.power(0.7, k) for k in range(dim_x)])
    x = np.random.multivariate_normal(np.zeros(dim_x), cov_mat, size=[n_obs, ])

    d = a_0 * x[:, 0] +   a_1 * np.divide(np.exp(x[:, 2]), 1 + np.exp(x[:, 2])) \
        + s_1 * np.random.standard_normal(size=[n_obs, ])
        
    y = alpha * d  + b_0 * np.divide(np.exp(x[:, 0]), 1 + np.exp(x[:, 0])) \
        + b_1 * x[:, 2] + s_2 * np.random.standard_normal(size=[n_obs, ])
    
    # add non-linearity in data generating process of y:
    if add_additional_nonlinearity:
        y += x[:, 3]*x[:, 4] +  3*x[:, 5]*(x[:, 5]>0)
    
    # print information about explained and unexplained variation in d and correlation of d and x variables:
    if 0:
        print('Ratio of unexplained variation to all variation in d:')
        print('s_1/np.var(d): %.3f'%(s_1/np.var(d)))
        print('Correlation of d and first 3 variables in x:')
        print(np.round( pd.DataFrame(np.column_stack((d,x[:,:3]))).corr() ,2))
        
    if return_type in _array_alias:
        return x, y, d
    elif return_type in _data_frame_alias + _dml_data_alias:
        x_cols = [f'X{i + 1}' for i in np.arange(dim_x)]
        data = pd.DataFrame(np.column_stack((x, y, d)),
                            columns=x_cols + ['y', 'd'])
        if return_type in _data_frame_alias:
            return data
        else:
            return DoubleMLData(data, 'y', 'd', x_cols)
    else:
        raise ValueError('Invalid return_type.')
        

    
        
def make_irm_data_II(n_obs=500, dim_x=20, theta=0, R2_d=0.5, R2_y=0.5, return_type='DoubleMLData', add_additional_nonlinearity = False):
    """
    Modified version of make_irm_data from DoubleML package.
    Generates data from a interactive regression (IRM) model.
    The data generating process is defined as

    .. math::

        d_i &= 1\\left\\lbrace \\frac{\\exp(c_d x_i' \\beta)}{1+\\exp(c_d x_i' \\beta)} > v_i \\right\\rbrace, & &v_i
        \\sim \\mathcal{U}(0,1),

        y_i &= \\theta d_i + c_y x_i' \\beta d_i + \\zeta_i, & &\\zeta_i \\sim \\mathcal{N}(0,1),

    with covariates :math:`x_i \\sim \\mathcal{N}(0, \\Sigma)`, where  :math:`\\Sigma` is a matrix with entries
    :math:`\\Sigma_{kj} = 0.5^{|j-k|}`.
    :math:`\\beta` is a `dim_x`-vector with entries :math:`\\beta_j=\\frac{1}{j^2}` and the constants :math:`c_y` and
    :math:`c_d` are given by

    .. math::

        c_y = \\sqrt{\\frac{R_y^2}{(1-R_y^2) \\beta' \\Sigma \\beta}}, \\qquad c_d =
        \\sqrt{\\frac{(\\pi^2 /3) R_d^2}{(1-R_d^2) \\beta' \\Sigma \\beta}}.

    The data generating process is inspired by a process used in the simulation experiment (see Appendix P) of Belloni
    et al. (2017).

    Parameters
    ----------
    n_obs :
        The number of observations to simulate.
    dim_x :
        The number of covariates.
    theta :
        The value of the causal parameter.
    R2_d :
        The value of the parameter :math:`R_d^2`.
    R2_y :
        The value of the parameter :math:`R_y^2`.
    return_type :
        If ``'DoubleMLData'`` or ``DoubleMLData``, returns a ``DoubleMLData`` object.

        If ``'DataFrame'``, ``'pd.DataFrame'`` or ``pd.DataFrame``, returns a ``pd.DataFrame``.

        If ``'array'``, ``'np.ndarray'``, ``'np.array'`` or ``np.ndarray``, returns ``np.ndarray``'s ``(x, y, d)``.

    References
    ----------
    Belloni, A., Chernozhukov, V., Fernández‐Val, I. and Hansen, C. (2017). Program Evaluation and Causal Inference With
    High‐Dimensional Data. Econometrica, 85: 233-298.
    """
    # inspired by https://onlinelibrary.wiley.com/doi/abs/10.3982/ECTA12723, see suplement
    v = np.random.uniform(size=[n_obs, ])
    zeta = np.random.standard_normal(size=[n_obs, ])

    cov_mat = toeplitz([np.power(0.5, k) for k in range(dim_x)])
    x = np.random.multivariate_normal(np.zeros(dim_x), cov_mat, size=[n_obs, ])

    beta = [1 / (k**2) for k in range(1, dim_x + 1)]
    b_sigma_b = np.dot(np.dot(cov_mat, beta), beta)
    c_y = np.sqrt(R2_y/((1-R2_y) * b_sigma_b))
    c_d = np.sqrt(np.pi**2 / 3. * R2_d/((1-R2_d) * b_sigma_b))

    xx = np.exp(np.dot(x, np.multiply(beta, c_d)))
    d = 1. * ((xx/(1+xx)) > v)

    y = d * theta + d * np.dot(x, np.multiply(beta, c_y)) + zeta

    # add non-linearity in data generating process of y:
    if add_additional_nonlinearity:
        y += x[:, 3]*x[:, 4] +  3*x[:, 5]*(x[:, 5]>0)

    if return_type in _array_alias:
        return x, y, d
    elif return_type in _data_frame_alias + _dml_data_alias:
        x_cols = [f'X{i + 1}' for i in np.arange(dim_x)]
        data = pd.DataFrame(np.column_stack((x, y, d)),
                            columns=x_cols + ['y', 'd'])
        if return_type in _data_frame_alias:
            return data
        else:
            return DoubleMLData(data, 'y', 'd', x_cols)
    else:
        raise ValueError('Invalid return_type.')
        
        
        

def make_pliv_CHS2015_II(n_obs, alpha=1., dim_x=200, dim_z=150, return_type='DoubleMLData', add_additional_nonlinearity = False):
    """
    Modified version of make_pliv_CHS2015 from DoubleML package.
    Generates data from a partially linear IV regression model used in Chernozhukov, Hansen and Spindler (2015).
    The data generating process is defined as

    .. math::

        z_i &= \\Pi x_i + \\zeta_i,

        d_i &= x_i' \\gamma + z_i' \\delta + u_i,

        y_i &= \\alpha d_i + x_i' \\beta + \\varepsilon_i,

    with

    .. math::

        \\left(\\begin{matrix} \\varepsilon_i \\\\ u_i \\\\ \\zeta_i \\\\ x_i \\end{matrix} \\right) \\sim
        \\mathcal{N}\\left(0, \\left(\\begin{matrix} 1 & 0.6 & 0 & 0 \\\\ 0.6 & 1 & 0 & 0 \\\\
        0 & 0 & 0.25 I_{p_n^z} & 0 \\\\ 0 & 0 & 0 & \\Sigma \\end{matrix} \\right) \\right)

    where  :math:`\\Sigma` is a :math:`p_n^x \\times p_n^x` matrix with entries
    :math:`\\Sigma_{kj} = 0.5^{|j-k|}` and :math:`I_{p_n^z}` is the :math:`p_n^z \\times p_n^z` identity matrix.
    :math:`\\beta = \\gamma` is a :math:`p_n^x`-vector with entries :math:`\\beta_j=\\frac{1}{j^2}`,
    :math:`\\delta` is a :math:`p_n^z`-vector with entries :math:`\\delta_j=\\frac{1}{j^2}`
    and :math:`\\Pi = (I_{p_n^z}, 0_{p_n^z \\times (p_n^x - p_n^z)})`.

    Parameters
    ----------
    n_obs :
        The number of observations to simulate.
    alpha :
        The value of the causal parameter.
    dim_x :
        The number of covariates.
    dim_z :
        The number of instruments.
    return_type :
        If ``'DoubleMLData'`` or ``DoubleMLData``, returns a ``DoubleMLData`` object.

        If ``'DataFrame'``, ``'pd.DataFrame'`` or ``pd.DataFrame``, returns a ``pd.DataFrame``.

        If ``'array'``, ``'np.ndarray'``, ``'np.array'`` or ``np.ndarray``, returns ``np.ndarray``'s ``(x, y, d, z)``.

    References
    ----------
    Chernozhukov, V., Hansen, C. and Spindler, M. (2015), Post-Selection and Post-Regularization Inference in Linear
    Models with Many Controls and Instruments. American Economic Review: Papers and Proceedings, 105 (5): 486-90.
    """
    assert dim_x >= dim_z
    # see https://assets.aeaweb.org/asset-server/articles-attachments/aer/app/10505/P2015_1022_app.pdf
    xx = np.random.multivariate_normal(np.zeros(2),
                                       np.array([[1., 0.6], [0.6, 1.]]),
                                       size=[n_obs, ])
    epsilon = xx[:, 0]
    u = xx[:, 1]

    sigma = toeplitz([np.power(0.5, k) for k in range(0, dim_x)])
    x = np.random.multivariate_normal(np.zeros(dim_x),
                                      sigma,
                                      size=[n_obs, ])

    I_z = np.eye(dim_z)
    xi = np.random.multivariate_normal(np.zeros(dim_z),
                                       0.25*I_z,
                                       size=[n_obs, ])

    beta = [1 / (k**2) for k in range(1, dim_x + 1)]
    gamma = beta
    delta = [1 / (k**2) for k in range(1, dim_z + 1)]
    Pi = np.hstack((I_z, np.zeros((dim_z, dim_x-dim_z))))

    z = np.dot(x, np.transpose(Pi)) + xi
    d = np.dot(x, gamma) + np.dot(z, delta) + u
    y = alpha * d + np.dot(x, beta) + epsilon

    # add non-linearity in data generating process of y:
    if add_additional_nonlinearity:
        y += x[:, 3]*x[:, 4] +  3*x[:, 5]*(x[:, 5]>0)
        
        
    if return_type in _array_alias:
        return x, y, d, z
    elif return_type in _data_frame_alias + _dml_data_alias:
        x_cols = [f'X{i + 1}' for i in np.arange(dim_x)]
        z_cols = [f'Z{i + 1}' for i in np.arange(dim_z)]
        data = pd.DataFrame(np.column_stack((x, y, d, z)),
                            columns=x_cols + ['y', 'd'] + z_cols)
        if return_type in _data_frame_alias:
            return data
        else:
            return DoubleMLData(data, 'y', 'd', x_cols, z_cols)
    else:
        raise ValueError('Invalid return_type.')

        
        

def make_iivm_data_II(n_obs=500, dim_x=20, theta=1., alpha_x=0.2, return_type='DoubleMLData',  add_additional_nonlinearity = False):
    """
    Modified version of make_iivm_data from DoubleML package.
    Generates data from a interactive IV regression (IIVM) model.
    The data generating process is defined as

    .. math::

        d_i &= 1\\left\\lbrace \\alpha_x Z + v_i > 0 \\right\\rbrace,

        y_i &= \\theta d_i + x_i' \\beta + u_i,

    with :math:`Z \\sim \\text{Bernoulli}(0.5)` and

    .. math::

        \\left(\\begin{matrix} u_i \\\\ v_i \\end{matrix} \\right) \\sim
        \\mathcal{N}\\left(0, \\left(\\begin{matrix} 1 & 0.3 \\\\ 0.3 & 1 \\end{matrix} \\right) \\right).

    The covariates :math:`x_i \\sim \\mathcal{N}(0, \\Sigma)`, where  :math:`\\Sigma` is a matrix with entries
    :math:`\\Sigma_{kj} = 0.5^{|j-k|}` and :math:`\\beta` is a `dim_x`-vector with entries
    :math:`\\beta_j=\\frac{1}{j^2}`.

    The data generating process is inspired by a process used in the simulation experiment of Farbmacher, Gruber and
    Klaaßen (2020).

    Parameters
    ----------
    n_obs :
        The number of observations to simulate.
    dim_x :
        The number of covariates.
    theta :
        The value of the causal parameter.
    alpha_x :
        The value of the parameter :math:`\\alpha_x`.
    return_type :
        If ``'DoubleMLData'`` or ``DoubleMLData``, returns a ``DoubleMLData`` object.

        If ``'DataFrame'``, ``'pd.DataFrame'`` or ``pd.DataFrame``, returns a ``pd.DataFrame``.

        If ``'array'``, ``'np.ndarray'``, ``'np.array'`` or ``np.ndarray``, returns ``np.ndarray``'s ``(x, y, d, z)``.

    References
    ----------
    Farbmacher, H., Guber, R. and Klaaßen, S. (2020). Instrument Validity Tests with Causal Forests. MEA Discussion
    Paper No. 13-2020. Available at SSRN: http://dx.doi.org/10.2139/ssrn.3619201.
    """
    # inspired by https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3619201
    xx = np.random.multivariate_normal(np.zeros(2),
                                       np.array([[1., 0.3], [0.3, 1.]]),
                                       size=[n_obs, ])
    u = xx[:, 0]
    v = xx[:, 1]

    cov_mat = toeplitz([np.power(0.5, k) for k in range(dim_x)])
    x = np.random.multivariate_normal(np.zeros(dim_x), cov_mat, size=[n_obs, ])

    beta = [1 / (k**2) for k in range(1, dim_x + 1)]

    z = np.random.binomial(p=0.5, n=1, size=[n_obs, ])
    d = 1. * (alpha_x * z + v > 0)

    y = d * theta + np.dot(x, beta) + u

    # add non-linearity in data generating process of y:
    if add_additional_nonlinearity:
        y += x[:, 3]*x[:, 4] +  3*x[:, 5]*(x[:, 5]>0)
        
    if return_type in _array_alias:
        return x, y, d, z
    elif return_type in _data_frame_alias + _dml_data_alias:
        x_cols = [f'X{i + 1}' for i in np.arange(dim_x)]
        data = pd.DataFrame(np.column_stack((x, y, d, z)),
                            columns=x_cols + ['y', 'd', 'z'])
        if return_type in _data_frame_alias:
            return data
        else:
            return DoubleMLData(data, 'y', 'd', x_cols, 'z')
    else:
        raise ValueError('Invalid return_type.')

        
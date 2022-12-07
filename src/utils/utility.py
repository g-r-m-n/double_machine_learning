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
from xgboost import XGBClassifier,XGBRegressor
from sklearn.linear_model import Lasso, LogisticRegression 
from sklearn.neural_network import MLPClassifier, MLPRegressor 
from sklearn.preprocessing import StandardScaler


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
    
    def __init__(self, model_type, algo_type='RF', n_fold = 2, param_grids_reg=[],param_grids_class=[]):
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
                learner_reg   = XGBRegressor(n_estimators=100, max_depth=5,  learning_rate =0.1)   
                learner_class = XGBClassifier(n_estimators=100, max_depth=5,  learning_rate =0.1) 

            elif self.algo_type == 'Lasso':   
                learner_reg   = Lasso(alpha=0.1)   
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
                
                #ml learner for the nuisance function m0(X)=E[D|X]:
                if len(np.unique(data['d'])) <= 10:  
                    self.ml_m = clone(learner_class)
                    self.param_grids['ml_m'] = self.param_grids_class
                else:                    
                    self.ml_m = clone(learner_reg)
                    self.param_grids['ml_m'] = self.param_grids_reg
                
                self.model_obj  = dml.DoubleMLPLR(obj_dml_data, self.ml_l, self.ml_m, n_folds = self.n_folds)

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
                
                self.model_obj = dml.DoubleMLPLIV(obj_dml_data, self.ml_l, self.ml_m, self.ml_r, n_folds = self.n_folds)
                
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
        if self.type_dml:
           self.model_obj.tune(self.param_grids, n_folds_tune = self.n_folds, **kwargs)
         
       
    def fit(self, **kwargs):
        self.fitted_model = self.model_obj.fit(**kwargs)
        if self.type_dml:
            self.fitted_model.interval = self.model_obj.confint()
        if self.type_lreg :
            self.fitted_model.interval = self.fitted_model.conf_int()   
            self.fitted_model.coef = self.fitted_model.params
            
    def collect_results(self,results_rep,i_rep): 
        results_rep.loc[i_rep,'coef_'+self.type+' '+self.algo_type] = self.fitted_model.coef[0] 
        results_rep.loc[i_rep,'lower_bound_'+self.type+' '+self.algo_type] = self.fitted_model.interval.iloc[0,0] 
        results_rep.loc[i_rep,'upper_bound_'+self.type+' '+self.algo_type] = self.fitted_model.interval.iloc[0,1] 
        return  results_rep
    


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


def get_res_stats(results_rep, model_index, theta, PRINT =1):
    if 0:#len(results_rep)<= 1:
        return []
    res_stats = pd.DataFrame()
    for m in model_index:
        model_type = m.split(' ')[0].lower()
        algo_type =  m.split(' ')[1] if len(m.split(' '))>1 else ''
        res_stats = pd.concat( [res_stats, pd.DataFrame([error_stats(theta, results_rep['coef_'+model_type+' '+algo_type])],index=[m])], axis=0 )
        
    # add best performing model:
    res_stats2 = res_stats.copy()
    if 'Bias' in res_stats2.columns:
   	    res_stats2['Bias'] = abs(res_stats2['Bias'])
    res_stats.loc['Best',:] =  res_stats2.idxmin()
    if PRINT:
        print(res_stats)
    return res_stats


def non_orth_score_w_g(y, d, l_hat, m_hat, g_hat, smpls, *kwargs):
    """ author: DoubleML python package.
    Non-orthotogonal score for partial linear model, i.e., the naive partial linear model. """
    u_hat = y - g_hat
    psi_a = -np.multiply(d, d)
    psi_b = np.multiply(d, u_hat)
    return psi_a, psi_b



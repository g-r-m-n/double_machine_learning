# %% IV model

import numpy as np

import doubleml as dml

from doubleml.datasets import make_pliv_CHS2015

from sklearn.ensemble import RandomForestRegressor

from sklearn.base import clone

learner = RandomForestRegressor(n_estimators=100, max_features=20, max_depth=5, min_samples_leaf=2)

ml_l = clone(learner)

ml_m = clone(learner)

ml_r = clone(learner)

np.random.seed(2222)

data = make_pliv_CHS2015(alpha=0.5, n_obs=500, dim_x=20, dim_z=1, return_type='DataFrame')

obj_dml_data = dml.DoubleMLData(data, 'y', 'd', z_cols='Z1')

dml_pliv_obj = dml.DoubleMLPLIV(obj_dml_data, ml_l, ml_m, ml_r)

print(dml_pliv_obj.fit())



#%% standard IV
#from linearmodels import IV2SLS, IVGMM, OLS
from statsmodels.sandbox.regression.gmm import IV2SLS  

exog = [i for i in data.columns if i.startswith('X')]

resultIV2SLS = IV2SLS(endog=data.loc[:,'y'], exog = data.loc[:,['d']+exog], instrument= data.loc[:,exog+['Z1']]).fit()
print(resultIV2SLS.summary())



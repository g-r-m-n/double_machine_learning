# %% Setup
# Install DoubleML package
# pip install -U DoubleML

# Website
# https://docs.doubleml.org/stable/index.html

import numpy as np

from doubleml.datasets import fetch_bonus, DoubleMLData

from doubleml import DoubleMLPLR

from sklearn.base import clone

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LassoCV


df_bonus = fetch_bonus('DataFrame')

print(df_bonus.head(5))


np.random.seed(3141)

n_obs = 500

n_vars = 100

theta = 3


X = np.random.normal(size=(n_obs, n_vars))

d = np.dot(X[:, :3], np.array([5, 5, 5])) + np.random.standard_normal(size=(n_obs,))

y = theta * d + np.dot(X[:, :3], np.array([5, 5, 5])) + np.random.standard_normal(size=(n_obs,))


# %% specify model

dml_data_bonus = DoubleMLData(df_bonus,
                y_col='inuidur1',
                d_cols='tg',
                x_cols=['female', 'black', 'othrace', 'dep1', 'dep2',
                        'q2', 'q3', 'q4', 'q5', 'q6', 'agelt35', 'agegt54',
                        'durable', 'lusd', 'husd'])
 

print(dml_data_bonus)




dml_data_sim = DoubleMLData.from_arrays(X, y, d)

print(dml_data_sim)



# %% Learners to estimate the nuisance models

learner = RandomForestRegressor(n_estimators = 500, max_features = 'sqrt', max_depth= 5)

ml_l_bonus = clone(learner)

ml_m_bonus = clone(learner)

learner = LassoCV()

ml_l_sim = clone(learner)

ml_m_sim = clone(learner)



#%% Estimate double/debiased machine learning models


# bonus data
np.random.seed(3141)

obj_dml_plr_bonus = DoubleMLPLR(dml_data_bonus, ml_l_bonus, ml_m_bonus)

obj_dml_plr_bonus.fit();

print(obj_dml_plr_bonus)





# using the simulated data
obj_dml_plr_sim = DoubleMLPLR(dml_data_sim, ml_l_sim, ml_m_sim)

obj_dml_plr_sim.fit();

print(obj_dml_plr_sim)






import doubleml as dml
import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LassoCV, LinearRegression

np.random.seed(1234)
n_obs  = 500
n_vars = 200
X = np.random.normal(size=(n_obs, n_vars))
theta = np.array([3.])
y = np.dot(X[:, :1], theta) + np.random.standard_normal(size=(n_obs,))
dml_data = dml.DoubleMLData.from_arrays(X[:, 1:], y, X[:, :1])

learner = LassoCV()
ml_l = clone(learner)
ml_m = clone(learner)
ml_g = clone(learner)

dml_plr = dml.DoubleMLPLR(dml_data, ml_l, ml_m, ml_g, score='IV-type')

dml_plr.fit(store_predictions = True)
print(dml_plr.summary)

# The predictions are then stored in an numpy array with dimensions (number of observations x number of cross-fitting repetitions x number of treatment variables). You can access these arrays to calculate and plot the residuals

# Predictions for nuisance part 'ml_l' and 'ml_m' stored in an array with dimensions (n_obs x n_rep x n_treat)
print(dml_plr.predictions['ml_l'].shape)
print(dml_plr.predictions['ml_m'].shape)


dml_plr.fitted_y =  dml_data.data['d'] * dml_plr.all_coef[0]  + dml_plr.predictions['ml_g'][:,:,0].flatten()


print('\nDML MAE:  %0.3f'%np.mean(abs(dml_data.y - dml_plr.fitted_y)))
print('DML RMSE: %0.3f'%np.sqrt(np.mean((dml_data.y - dml_plr.fitted_y)**2)))



lin_reg = LinearRegression()
lin_reg.fit(dml_data.data[dml_data.d_cols+dml_data.x_cols],dml_data.y)
print('\nOLS estimate of treatment effect: %0.3f'%lin_reg.coef_[:1])
lin_reg.fitted_y = lin_reg.predict(dml_data.data[dml_data.d_cols+dml_data.x_cols])

print('\nOLS MAE:  %0.3f'%np.mean(abs(dml_data.y - lin_reg.fitted_y)))
print('OLS RMSE: %0.3f'%np.sqrt(np.mean((dml_data.y - lin_reg.fitted_y)**2)))

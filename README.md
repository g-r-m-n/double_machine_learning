# DML - Double Machine Learning Experimentation


## Motivation
The double/debiased machine learning (DML) model is increasing in popularity as it is designed to use machine learning for the analysis of causal effects.
We compare the performance of four variations of the DML model with a naive machine learning model and simple linear regression models. 
For doing so, we compare the estimation results for four different simulation experiment scenarios implemented in the Python package [DoubleML](https://github.com/DoubleML) from Bach et al. 2022. 
The simulation experiment scenarios are either with linear or non-linear effects and with or without endogeneity of the causal effect variable.

The considered machine learning algorithmes are Random Forest, Lasso, XGBoost and Neural Nets.

## Setup
See the DML/requirements.txt file for package requirements to run the analysis.


## Run
Adjust the settings in the first junk of the script DML/src/scenario_run.py or leave the default settings. Run the script DML/src/scenario_run.py. It conducts per default all experimentation scenario runs, including data generation and tuning if wanted, and produces as result summary tables and plots of the outcomes.


## References
Bach, P., Chernozhukov, V., Kurz, M. S., and Spindler, M., DoubleML - An Object-Oriented Implementation
230 of Double Machine Learning in Python, Journal of Machine Learning Research, 23(53): 1-6, 2022.

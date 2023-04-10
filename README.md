# DML - Double Machine Learning Experimentation


## Motivation
The double/debiased machine learning (DML) model is increasing in popularity as it is designed to use machine learning for the analysis of causal effects.
We compare the performance of four variations of the DML model with a naive machine learning model and simple linear regression models. 
For doing so, we compare the estimation results for four different simulation experiment scenarios implemented in the Python package [DoubleML](https://github.com/DoubleML) from Bach et al. 2022. 
The simulation experiment scenarios are either with linear or non-linear effects and with or without endogeneity of the causal effect variable.

The considered machine learning algorithmes include Random Forest, Lasso, XGBoost and Neural Nets.


## Setup
See the DML/requirements.txt file for package requirements to run the analysis.


## Run
Specify the settings in a dedicated .yaml file and save it to the folder ./src/config or use the default settings. Per default the script uses the specifications from default_config.yaml. Run the script DML/src/scenario_run.py. It conducts in the default setting all experimentation scenario runs, including data generation and tuning if wanted, and produces as result summary tables (in .csv and .tex formats) and plots (in .png and .pdf formats) of the outcomes.

## Summary
A paper that summarizes the approach, outcomes and findings is available [here](https://github.com/g-r-m-n/dml/blob/main/Is%20Double%20Machine%20Learning%20always%20better%20than%20Simple%20Linear%20Regression%20to%20estimate%20Causal%20Effects%20-%20Evidence%20from%20four%20simulation%20experiments.pdf).


## References
Bach, P., Chernozhukov, V., Kurz, M. S., and Spindler, M., DoubleML - An Object-Oriented Implementation
230 of Double Machine Learning in Python, Journal of Machine Learning Research, 23(53): 1-6, 2022.

# Uncovering differential equations from data with hidden variables

This repository has all the code implementation of the paper: 
[Uncovering differential equations from data with hidden variables](https://arxiv.org/abs/2002.02250)

### Abstract
SINDy is a method for learning system of differential equations from data by solving a sparse 
linear regression optimization problem. In this article, we propose an extension of the SINDy 
method that learns systems of differential equations in cases where some of the variables are 
not observed. Our extension is based on regressing a higher order time derivative of a target 
variable onto a dictionary of functions that includes lower order time derivatives of the target 
variable. We evaluate our method by measuring the prediction accuracy of the learned dynamical 
systems on synthetic data and on a real data-set of temperature time series provided by the Réseau 
de Transport d'Électricité (RTE). Our method provides high quality short-term forecasts and it is 
orders of magnitude faster than competing methods for learning differential equations with latent 
variables.

### Examples
The experiments performed in the paper can be found in the notebook examples/paper_experiments.ipynb. 
There you will find:
1. Experiments with all variables observed.
2. Experiments with only one variable observed:
    1. Oscilator variable x
    1. Rössler variable y
    1. Rössler variable x
    1. Lorenz variable x
3. Example of RTE experimental setting.

We also provide in examples/spatial_experiments some insights on prelimiar experiments using this 
methodology to solve PDE where some variable is not observed. In the example, a unidimentional
wave equation with varying (and unknown) tension is used to generate data. 



# Setup for developers
Create virtual enviroment
```
python3.8 -m venv venv
```

Activate virtual enviroment
```
. .venv/bin/activate
```
Install libraries 
```
pip install -r requirements.txt 
```

### Jupyter notebooks
In order to be able to run jupyter notebooks:
```
pip install ipykernel
python -m ipykernel install --user --name venv --display-name "venv"
```
Source: https://ipython.readthedocs.io/en/stable/install/kernel_install.html#kernels-for-different-environments 

   
"""
Nonparametric Econometrics Problem set 3 Problem 1. Author: Mate Kormos
"""

# import dependencies
import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy.stats as sstat
from sklearn import preprocessing

# get the data
path = r'C:\Users\Máté\Dropbox\CEU\2017 Spring\Nonparametric\Nonparametric_ps3\mroz.xlsx'
data = pd.read_excel(path, index_col=None)
# size
n = len(data)

############################################## PART A ########################################################
print('\n####################################################\n PART A \n #########################################\n')
# outcome equation estimated with OLS
# y
y = data['lwage']
# regressors
x = data[['educ', 'exper', 'expersq']].values
# constant
constant = np.ones((n, 1))
# design matrix
X = np.concatenate((x, constant), axis=1)
# regression
results = sm.OLS(endog=y, exog=X, missing='drop', hasconst=True).fit()
print('OLS results:\n', results.summary(xname=['educ', 'exper', 'expersq', 'const'], yname='lwage'))


############################################## PART B ########################################################
print('\n####################################################\n PART B \n #########################################\n')
# outcome equation estimated with parametric Heckit

# First stage
# d
d = data['inlf'].values
z = data[['educ', 'exper', 'expersq', 'age', 'kidslt6', 'kidsge6', 'nwifeinc']].values
Z = np.concatenate((z, constant), axis=1)
# first stage, z'*delta score from probit of D on constant and z (that is, on Z)
# estimate the coeffs
deltahat = sm.Probit(endog=d, exog=Z).fit().params
# compute the z'*delta score for each observation
score = Z.dot(deltahat)
# create a function which produces lambdahat


def lambdahat_fn(score_in):
    numerator = np.array(list(map(sstat.norm.pdf, score_in))).T
    denominator = np.array(list(map(sstat.norm.cdf, score_in))).T
    lambda_est = numerator / denominator
    return lambda_est


# lambdahat
lambdahat = lambdahat_fn(score_in=score)

# Second stage
# design matrix with lambdahat included
X_b = np.concatenate((X, lambdahat[:, None]), axis=1)
# estimate with OLS
results = sm.OLS(endog=y, exog=X_b, missing='drop', hasconst=True).fit()
print('Parametric Heckit results:\n', results.summary(xname=['educ', 'exper', 'expersq', 'const', 'lambdhahat'],
                                                      yname='lwage'))


############################################## PART C ########################################################
print('\n####################################################\n PART C \n #########################################\n')
# nonparametric Heckit

# First stage
# create interactions and powers
# degree
K = 2
# drop 'expersq' from z
z_c = data[['educ', 'exper', 'age', 'kidslt6', 'kidsge6', 'nwifeinc']].values
# interactions and squares
Z_series = preprocessing.PolynomialFeatures(degree=K, include_bias=True).fit_transform(z_c)
# choose series model
model = 'logit'
if model == 'OLS':
    seriesmodel = sm.OLS(endog=d, exog=Z_series, hasconst=True)
elif model == 'logit':
    seriesmodel = sm.Logit(endog=d, exog=Z_series, hasconst=True)
# fit series model
results = seriesmodel.fit(maxiter=1000)
print('\nNonparametric Heckit, first stage series model:\n', results.summary(yname='d'))
# predict propensity score
pscorehat = seriesmodel.predict(exog=Z_series, params=results.params)

# Second stage
pscorehat_vector = pscorehat[:, None]
X_c = np.concatenate((X, pscorehat_vector, pscorehat_vector ** 2, pscorehat_vector ** 3, pscorehat_vector ** 4,
                      pscorehat_vector ** 5), axis=1)
results = sm.OLS(endog=y, exog=X_c, missing='drop', hasconst=True).fit()
print('\n\n ############### \n Heckit second stage OLS, (first coeff is educ):\n', results.summary(yname='lwage'))

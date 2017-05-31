# Nonparametric Econometrics Problem set 2 Problem 3 Part (a). Author: Mate Kormos

# Import dependencies
import numpy as np

# Reading the files
# black
with open(r'C:\Users\Máté\Dropbox\CEU\2017 Spring\Nonparametric\Nonparametric_ps2\bweight_age_black_20K.txt', 'r')\
          as myfile:
    filein = myfile.readlines()
# break the lines and convert into np array
rawdata = np.array([float(line.split()[0]) for line in filein])
# weight
weight_b = rawdata[0:int(len(rawdata) / 2)]
# age
age_b = rawdata[int(len(rawdata) / 2):]
# white
with open(r'C:\Users\Máté\Dropbox\CEU\2017 Spring\Nonparametric\Nonparametric_ps2\bweight_age_white_20K.txt', 'r')\
          as myfile:
    filein = myfile.readlines()
# break the lines and convert into np array
rawdata = np.array([float(line.split()[0]) for line in filein])
# weight
weight_w = rawdata[0:int(len(rawdata) / 2)]
# age
age_w = rawdata[int(len(rawdata) / 2):]

# PART (a) #
# OLS regression and significance tests
# define dependent
y_b = np.matrix(weight_b).T
y_w = np.matrix(weight_w).T
# define design matrix
X_b = np.matrix(age_b).T
X_w = np.matrix(age_w).T
# add constant
X_b = np.insert(X_b, 0, 1, axis=1)
X_w = np.insert(X_w, 0, 1, axis=1)
# estimate the coeffs
X_bT = X_b.T
X_wT = X_w.T
X_bTX_bTinv = np.linalg.inv(X_bT * X_b)
X_wTX_wTinv = np.linalg.inv(X_wT * X_w)
bhat_b = X_bTX_bTinv * X_bT * y_b
bhat_w = X_wTX_wTinv * X_wT * y_w
# print coeffs
print('bhat_b = \n', bhat_b, '\n')
print('bhat_w = \n', bhat_w, '\n')
# residuals
n = len(y_b)
uhat_b = y_b - X_b * bhat_b
print('Residual_b estimation done\n')
uhat_w = y_w - X_w * bhat_w
print('Residual_w estimation done\n')
# estimate White's cov matrix by parts report on progress
covhat_b_p1 = X_bTX_bTinv * X_bT * np.diag(np.array(uhat_b.flatten())[0] ** 2)
print('covhat_b part 1 done\n')
covhat_b_p2 = X_b * X_bTX_bTinv
print('covhat_b part 2 done\n')
covhat_b = covhat_b_p1 * covhat_b_p2
print('covhat_b done\n')
covhat_w_p1 = X_wTX_wTinv * X_wT * np.diag(np.array(uhat_w.flatten())[0] ** 2)
print('covhat_w part 1 done\n')
covhat_w_p2 = X_w * X_wTX_wTinv
print('covhat_b part 2 done\n')
covhat_w = covhat_w_p1 * covhat_w_p2
print('covhat_w done\n')
# SEs
sehat_b = np.sqrt(np.array(covhat_b[1])[0][1])
sehat_w = np.sqrt(np.array(covhat_w[1])[0][1])
print('White\'s cov_b matrix = \n', covhat_b, '\n')
print('White\'s cov_w matrix = \n', covhat_w, '\n')
# compute test statistics and print it
z_b = np.array(bhat_b[1])[0][0] / sehat_b
z_w = np.array(bhat_w[1])[0][0] / sehat_w
print('\n z_b = ', z_b, '\n')
print('\n z_w = ', z_w, '\n')
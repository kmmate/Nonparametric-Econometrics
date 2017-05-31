# Nonparametric econometrics Problem set 2 Problem 4. Author: Mate Kormos

# Import dependencies
import numpy as np
from nonparaecon import kde, npregression
import time

# Start timer
start_time = time.time()

############# DATA #############################
# Get the data
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

# Size
n = len(age_b)


# Random subsample and partitioning
np.random.seed([0])
trainsize = 500
testsize = 2000
permindex = np.random.permutation(n)
# train
age_b_train = age_b[permindex[0:trainsize]]
age_w_train = age_w[permindex[0:trainsize]]
weight_b_train = weight_b[permindex[0:trainsize]]
weight_w_train = weight_w[permindex[0:trainsize]]
#test
age_b_test = age_b[permindex[trainsize:trainsize + testsize]]
age_w_test = age_w[permindex[trainsize:trainsize + testsize]]
weight_b_test = weight_b[permindex[trainsize:trainsize + testsize]]
weight_w_test = weight_w[permindex[trainsize:trainsize + testsize]]

# Standardise data
# train
age_b_train_st = (age_b_train - age_b_train.mean()) / age_b_train.std()
age_w_train_st = (age_w_train - age_w_train.mean()) / age_w_train.std()
weight_b_train_st = (weight_b_train - weight_b_train.mean()) / weight_b_train.std()
weight_w_train_st = (weight_w_train - weight_w_train.mean()) / weight_w_train.std()
# test
age_b_test_st = (age_b_test - age_b_test.mean()) / age_b_test.std()
age_w_test_st = (age_w_test - age_w_test.mean()) / age_w_test.std()
weight_b_test_st = (weight_b_test - weight_b_test.mean()) / weight_b_test.std()
weight_w_test_st = (weight_w_test - weight_w_test.mean()) / weight_w_test.std()

# Time of preparing data
start_time_lscv_b = time.time()
print('Preparing data: %s seconds\n' % (start_time_lscv_b - start_time))

# ###################### TRAINING ##############################
# Bandwidth
kerneltype = 'epanechnikov'
h_age_b = kde.kde_pdf(x=None, sampledata=age_b_train_st, kerneltype=kerneltype, getsilverman=True)
h_age_w = kde.kde_pdf(x=None, sampledata=age_w_train_st, kerneltype=kerneltype, getsilverman=True)
h_weight_b = kde.kde_pdf(x=None, sampledata=weight_b_train_st, kerneltype=kerneltype, getsilverman=True)
h_weight_w = kde.kde_pdf(x=None, sampledata=weight_w_train_st, kerneltype=kerneltype, getsilverman=True)
print('Silvermans bandwidht, blacks, age and weight:', [h_age_b, h_weight_b])
print('Silvermans bandwidht, whites, age and weight:', [h_age_w, h_weight_w])
# run least squares cross validation to get the optimal bandwidth, give searchrange so that it includes Silverman's
# search ranges
searchrange_b = np.arange(0.6, 8, 0.5)
searchrange_w = np.arange(0.6, 8, 0.5)
# polynomial order
polorder = 2
# blacks
print('LSCV has started for blacks.')
hopt_b, sse_b = npregression.npregression_locpollscv(searchrange=searchrange_b, xdata=age_b_train_st,
                                                     ydata=weight_b_train_st,
                                                     polorder=polorder, kerneltype=kerneltype,
                                                     subsamplesize=None, get_SSElist=True)
# time of lscv
start_time_lscv_w = time.time()
print('LSCV, blacks: %s seconds\n' % (start_time_lscv_w - start_time_lscv_b))
print('hopt_b=', hopt_b, '\nsse_b=\n', sse_b)
print('LSCV has started for whites')
hopt_w, sse_w = npregression.npregression_locpollscv(searchrange=searchrange_w, xdata=age_w_train_st,
                                                     ydata=weight_w_train_st,
                                                     polorder=polorder, kerneltype=kerneltype,
                                                     subsamplesize=None, get_SSElist=True)
# time of lscv
start_time_testing = time.time()
print('LSCV, whites: %s seconds\n' % (start_time_testing - start_time_lscv_w))
print('hopt_w=', hopt_w, '\nsse_w=\n', sse_w)

############################################### TESTING ##############################
print('#####################################################\n TESTING \n#########################################\n')
# adjust optimal bandwidth for differences in size
d = 1
hopt_b_adjusted = hopt_b * (testsize / trainsize) ** (-1 / (2 * polorder + d + 2))
hopt_w_adjusted = hopt_w * (testsize / trainsize) ** (-1 / (2 * polorder + d + 2))
print('The sample-size adjusted optimal bandwidth, blacks:', hopt_b_adjusted)
print('The sample-size adjusted optimal bandwidth, whites:', hopt_w_adjusted)
# blacks
weighthat_test_b = np.array([npregression.npregression_locpol(x=i, xdata=age_b_test_st, ydata=weight_b_test_st,
                                                              polorder=polorder, kerneltype=kerneltype,
                                                              bandwidth=hopt_b_adjusted, getSE=False, getCI=False)
                    for i in age_b_test_st])
print('Testing done for blacks.')
# whites
weighthat_test_w =  np.array([npregression.npregression_locpol(x=i, xdata=age_w_test_st, ydata=weight_w_test_st,
                                                               polorder=polorder, kerneltype=kerneltype,
                                                               bandwidth=hopt_w_adjusted, getSE=False, getCI=False)
                    for i in age_w_test_st])

# compute SSE
MSSE_test_b = sum((weighthat_test_b - weight_b_test_st) ** 2) / testsize
MSSE_test_w = sum((weighthat_test_w - weight_w_test_st) ** 2) / testsize
print('Testing: mean sum of squared errors, blacks:', MSSE_test_b)
print('Testing: mean sum of squared errors, whites:', MSSE_test_w)

# time of testing
print('Testing: %s seconds\n' % (time.time() - start_time_testing))

############################## PLOTTING ##############################################
# bandwidth selections
npregression.npregression_plot(mhat=[sse_b, sse_w], ismultiple=True, xdomain=searchrange_b,
                               plottitle='Sum of squared errors from LSCV bandwidth selection, scalar $x$, $p=$'
                                         + str(polorder),
                               xlabel='$h$', ylabel='$SSE_{LSCV}(h)$', mtrueon=False, confidinton=False,
                               legendlabel=dict({'y_1': 'Blacks', 'y_2': 'Whites'}),
                               color=dict({'mhat_1': 'red', 'mhat_2': 'blue'}),
                               seriesperrow=2, savemode=True,
                               filepath=r'C:\Users\Máté\Dropbox\CEU\2017 Spring\Nonparametric\Nonparametric_ps2\LSCV',
                               viewmode=True)
# testing sample estimates
#blacks
npregression.npregression_plot(mhat=weighthat_test_b[np.argsort(age_b_test_st)], ismultiple=False,
                               xdomain=age_b_test_st[np.argsort(age_b_test_st)],
                               plottitle='Estimated birth weight, blacks, test sample, scalar $x$, $p=$'
                                         + str(polorder),
                               xlabel='Standardised age', ylabel='Standardised weight', mtrueon=True,
                               mtrue=weight_b_test_st[np.argsort(age_b_test_st)], truestyle='.',
                               confidinton=False, legendlabel=dict({'mhat': 'Estim.', 'mtrue': 'True'}),
                               color=dict({'mhat': 'blue', 'mtrue': 'red'}),
                               seriesperrow=2, savemode=True,
                               filepath=r'C:\Users\Máté\Dropbox\CEU\2017 Spring\Nonparametric\Nonparametric_ps2\testfit_b',
                               viewmode=True)
#whites
npregression.npregression_plot(mhat=weighthat_test_w[np.argsort(age_w_test_st)], ismultiple=False,
                               xdomain=age_w_test_st[np.argsort(age_w_test_st)],
                               plottitle='Estimated birth weight, whites, test sample, scalar $x$, $p=$'
                                         + str(polorder),
                               xlabel='Standardised age', ylabel='Standardised weight', mtrueon=True,
                               mtrue=weight_w_test_st[np.argsort(age_w_test_st)], truestyle='.',
                               confidinton=False, legendlabel=dict({'mhat': 'Estim.', 'mtrue': 'True'}),
                               color=dict({'mhat': 'blue', 'mtrue': 'red'}),
                               seriesperrow=2, savemode=True,
                               filepath=r'C:\Users\Máté\Dropbox\CEU\2017 Spring\Nonparametric\Nonparametric_ps2\testfit_w',
                               viewmode=True)

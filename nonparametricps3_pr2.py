"""
Nonparametric Econometrics Problem set 3 Problem 2. Author: Mate Kormos
"""

# import dependencies
import numpy as np
from tabulate import tabulate
from nonparaecon.npregression import npregression_rdd, npregression_rddlscv

# seed random
np.random.seed([0])
# settings
kerneltype = 'triangular'
kernelorder = 2
# local linear regression
poldegree = 1
# searchrange for optimal bandwidth
searchrange = np.arange(1, 4, 0.5)
# number of iteration in Monte Carlo
iternumber = 2000

# datagenerating process


def gen_x(n):
    return np.random.uniform(low=-2, high=2 + np.finfo(float).eps, size=n)


def gen_y(x, sigma):
    belowzero = int(x < 0)
    true = np.sin(x) * belowzero + np.cos(x) * (1 - belowzero)
    noise = np.random.normal(scale=sigma, size=1)[0]
    return true + noise

# matrix to print: columns: two sigma, three sample size; rows: the bias of tauhat1 and tauhat2, and the expected length
# and coverage probability of the three CI
# length of container vector
sum_vector_size = 2 + 2 * 3
# looping through sigma
for sigma in [0.3, 0.4]:
    # through n
    for n in [80, 160, 300]:
        print('\n #################\n Working on sigma= %.1f and n= %d' % (sigma, n), '#####\n')
        # container vector
        sum_vector = np.empty((sum_vector_size, ))
        # iterations
        for iteration in range(iternumber):
            # generate data
            x = gen_x(n=n)
            y = np.array([gen_y(x=i, sigma=sigma) for i in x])
            # optimal bandwidth
            h_opt = npregression_rddlscv(design='sharp', searchrange=searchrange, ydata=y, runningxdata=x, cutoff=0,
                                         treatmentabove=True, poldegree=poldegree)
            # linear regression
            tauhat_1, ci_11 = npregression_rdd(design='sharp', ydata=y, runningxdata=x, cutoff=0, poldegree=poldegree,
                                               treatmentabove=True, bandwidth=h_opt, get_ci=True)
            # quadratic with the same bandwidth
            tauhat_2, sehat_2, ci_22 = npregression_rdd(design='sharp', ydata=y, runningxdata=x, cutoff=0,
                                                        poldegree=poldegree + 1, treatmentabove=True, bandwidth=h_opt,
                                                        get_se=True, get_ci=True)
            # ci around tauhat_1 with tauhat_2 standard errors
            ci_12_low = tauhat_1 - 1.96 * sehat_2
            ci_12_high = tauhat_1 + 1.96 * sehat_2
            ci_12 = [ci_12_low, ci_12_high]
            # compute the statistics which will be averaged across the iterations
            # diff_1, will be bias(tauhat_1)
            diff_1 = tauhat_1 - 1
            # diff_2, will be bias(tauhat_2)
            diff_2 = tauhat_2 - 1
            # ci length
            ci_len_11 = ci_11[1] - ci_11[0]
            ci_len_12 = ci_12[1] - ci_12[0]
            ci_len_22 = ci_22[1] - ci_22[0]
            # whether includes true value, will be coverage probability
            istruecovered_11 = int(ci_11[0] <= 1 and 1 <= ci_11[1])
            istruecovered_12 = int(ci_12[0] <= 1 and 1 <= ci_12[1])
            istruecovered_22 = int(ci_22[0] <= 1 and 1 <= ci_22[1])
            # arrange into array and add to sum vector
            vectortoadd = np.array([diff_1, diff_2, ci_len_11, ci_len_12, ci_len_22, istruecovered_11, istruecovered_12,
                                    istruecovered_22])
            sum_vector += vectortoadd
        # average
        try:
            matrixtoprint = np.concatenate((matrixtoprint, (sum_vector / iternumber)[:, None]), axis=1)
        except (NameError, ValueError):
            matrixtoprint = np.empty((2 + 2 * 3, 1))
            matrixtoprint[:, 0] = sum_vector / iternumber
        print('As for the current sigma (%.1f) and n (%d)' % (sigma, n), 'the matrix is:\n', matrixtoprint)

# Print final
print('\n ###########################\n Final result: \n ', tabulate(tabular_data=matrixtoprint, tablefmt='latex',
                                                                     floatfmt='.4f'))
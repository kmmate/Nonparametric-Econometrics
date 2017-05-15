# Nonparametric Econometrics Problem set 1 Problem 2. Author: Mate Kormos

# Import dependencies
import numpy as np
from scipy import stats
from nonparaecon import kde
from tabulate import tabulate

# Number of iterations
M = 1000
# Preallocate arrays
# Is f(x) in {CIhat with Silverman's h}; 3 different x's, 4 different sample sizes, M iterations
isinconfid = np.empty((3, 4, M))
# Is f(x) in {CI with 0.25*(Silverman's h)}; 3 different x's, 4 different sample sizes, M iterations
isinconfid_s = np.empty((3, 4, M))
# probability coverage Silverman's h; 3 different x's, 4 different sample sizes, M iterations
probcov = np.empty((3, 4))
# probability coverage 0.25*(Silverman's h); 3 different x's, 4 different sample sizes, M iterations
probcov_s = np.empty((3,4))
# not bias-corrected fhat with Silverman's h; 3 different x's, 4 sample sizes, M iterations
fhat = np.empty((3, 4, M))
# Expected value of the not-bias corrected fhat with Silverman's h; 3 different x's, 4 sample sizes
Efhat = np.empty((3, 4))
# scaled bias with Silverman's h; 3 different x's, 4 sample sizes
scaled_bias = np.empty((3,4))
# Seed random
np.random.seed([0])

# For each x
# index
i_x = 0
# loop
for x in [1, 1.5, 2]:
    fx = stats.chi2.pdf(x=x, df=1)
    # for each sample size
    # index
    i_n = 0
    # loop
    for n in [50, 100, 250, 500]:
        print('Currently working on x=', x, ' and n=', n, '\n')
        # get Silverman's bandwidth, using the average of 40 samples of size n
        h_s = kde.kde_pdf(x=None, sampledata=np.mean([np.random.chisquare(df=1, size=n) for i in range(40)], 0),
                          kerneltype='epanechnikov', getsilverman=True)
        # iteration
        for m in range(M):
            # draw sample
            sample = np.random.chisquare(df=1, size=n)
            # estimate fhat and ci for Silverman and rescaled Silverman bandwidth
            # (for the coverage probability fhat is not need only the CI; fhat is needed only for
            # the rescaled bias computation)
            fhatt, ci_low, ci_high = kde.kde_pdf(x=x, sampledata=sample, kerneltype='epanechnikov',
                                             bandwidth=None, correctboundarybias=False,
                                             biascorrected=False, confidint=True)
            fhatt, ci_low_s, ci_high_s = kde.kde_pdf(x=x, sampledata=sample, kerneltype='epanechnikov',
                                                 bandwidth=None, bandwidthscale=0.25,
                                                 correctboundarybias=False, biascorrected=False,
                                                 confidint=True)
            fhat[i_x, i_n, m] = kde.kde_pdf(x=x, sampledata=sample, kerneltype='epanechnikov',
                                             bandwidth=None, correctboundarybias=False,
                                             biascorrected=False, confidint=False)
            # is the true f contained in the estimated ci? 1 if yes, 0 if not
            isinconfid[i_x, i_n, m] = int(ci_low <= fx <= ci_high)
            isinconfid_s[i_x, i_n, m] = int(ci_low_s <= fx <= ci_high_s)

        # compute the rescaled bias
        Efhat[i_x, i_n] = np.mean(fhat, 2)[i_x][i_n]
        scaled_bias[i_x, i_n] = np.sqrt(n * h_s) * (Efhat[i_x, i_n] - fx)

        # update index of n
        i_n = i_n + 1
    # update index of x
    i_x = i_x + 1

# Compute the probability coverage
probcov = np.mean(isinconfid, 2)
probcov_s = np.mean(isinconfid_s, 2)

# Print to latex table
# probability coverage
print('\n ##########\n Probability coverage:\n', tabulate(probcov, tablefmt='latex', floatfmt='.4f'), '\n#############')
# scaled probability coverage
print('\n ##########\n Probability coverage scaled:\n', tabulate(probcov_s, tablefmt='latex', floatfmt='.4f'),
      '\n############')
# rescalled bias
print('\n ######## Rescaled bias:\n ', tabulate(scaled_bias, tablefmt='latex', floatfmt='.4f'),
      '\n ############')
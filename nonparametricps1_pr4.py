# Nonparametric Econometrics Problem set 1 Problem 2. Author: Mate Kormos

# Import dependencies
import numpy as np
from nonparaecon import kde
from tabulate import tabulate

# Sample number
n = 100
# Iteration number
M = 10000
# Preallocate arrays
bias = np.empty((5, 5))  # 5 x's, (4 (bandwidth,kerneltype) tuples + 1 mean column)
fhat_m = np.empty((M, 5))  # fhats for the given iteration
Efhat = np.empty((5, 5))  # the approximated expected value for each x and (bandwidth, kerneltype) tuples
# True f(x) for f~uniform[0,1]
f = 1
# Seed random
np.random.seed([0])

# For each x
# index
i = 0
# loop
for x in [0, 0.05, 0.1, 0.5, 0.6]:
    # iteration
    for m in range(M):
        # draw sample of size n
        sample = np.random.rand(n)
        # compute the kernel density estimation, given the sample, for (bandwidth, kerneltype) tuples
        fhat_m[m, 0:4] = [kde.kde_pdf(x=x, sampledata=sample, bandwidth=0.1,
                                      kerneltype='epanechnikov', biascorrected=False),
                          kde.kde_pdf(x=x, sampledata=sample, bandwidth=0.25,
                                      kerneltype='epanechnikov', biascorrected=False),
                          kde.kde_pdf(x=x, sampledata=sample, bandwidth=0.1,
                                      kerneltype='gaussian', biascorrected=False),
                          kde.kde_pdf(x=x, sampledata=sample, bandwidth=0.25,
                                      kerneltype='epanechnikov', biascorrected=False)
                          ]
        # mean over (bandwidth,kerneltype) tuples
        fhat_m[m, 4] = np.mean(fhat_m[m, 0:4])
    # average over iterations
    Efhat[i, :] = np.mean(fhat_m, 0)
    # Compute bias
    bias[i, :] = Efhat[i,:] - f
    # increase row index for the next x
    i = i + 1

# Print bias to latex table
print(tabulate(bias, tablefmt='latex', floatfmt='.4f'))
# Nonparametric Econometrics Problem set 1 Problem 1. Author: Mate Kormos

# Import dependencies
import numpy as np
from nonparaecon import kde

# Read the files
# white
with open(r'C:\Users\Máté\Dropbox\CEU\2017 Spring\Nonparametric\Nonparametric_ps1\bweight_white_20K.txt', 'r') \
        as myfile:
    filein = myfile.readlines()
# break lines and convert into numpy array
bw_white = np.array([float(line.split()[0]) for line in filein])
# black
with open(r'C:\Users\Máté\Dropbox\CEU\2017 Spring\Nonparametric\Nonparametric_ps1\bweight_black_20K.txt', 'r') \
        as myfile:
    filein = myfile.readlines()
# break lines and convert into numpy array
bw_black = np.array([float(line.split()[0]) for line in filein])

# Part (a) #
# kernel density estimation for whites, Epa kernel, Silverman's bandwidth, plot
# domain
support = np.arange(0, bw_white.max() + 0.3 * bw_white.std(), 100)
# estimation
fhat_white_s = [kde.kde_pdf(x=x, sampledata=bw_white, kerneltype='epanechnikov', bandwidth=None, bandwidthscale=None,
                            biascorrected=False, correctboundarybias=False) for x in support]
# plot
kde.kde_plot(fhat=fhat_white_s, ismultiple=False, fsupport=support, plottitle='KDE of white birthweight, Epa. with '
             'Silverman\'s bandwidth', xlabel='gram', ylabel='$\hat{f}(x)$', savemode=True,
             filepath=r'C:\Users\Máté\Dropbox\CEU\2017 Spring\Nonparametric\Nonparametric_ps1\Problem1_a',
             viewmode=True)

# Part (b) #
# kernel density estimation for whites, Epa kernel, Silverman's bandwidth times 1/5 and 5, plot
# estimation
fhat_white_s_1over5 = [kde.kde_pdf(x=x, sampledata=bw_white, kerneltype='epanechnikov',
                                   bandwidth=None, bandwidthscale=1 / 5,
                                   biascorrected=False, correctboundarybias=False) for x in support]
fhat_white_s_5 = [kde.kde_pdf(x=x, sampledata=bw_white, kerneltype='epanechnikov',
                                   bandwidth=None, bandwidthscale=5,
                                   biascorrected=False, correctboundarybias=False) for x in support]
# plot
kde.kde_plot(fhat=[fhat_white_s, fhat_white_s_1over5, fhat_white_s_5], ismultiple=True,
             fsupport=support, plottitle='KDE of white birthweight, Epa. with '
             '(Silverman\'s bandwidth)$*j$, $j=1,0.2,5.$', xlabel='gram', ylabel='$\hat{f}(x)$',
             color={'fhat_1': 'red', 'fhat_2': 'royalblue', 'fhat_3': 'lime'},
             legendlabel={'dist_1': '$\hat{f}(x)$, $j=1$', 'dist_2': '$\hat{f}(x)$, $j=0.2$',
                          'dist_3': '$\hat{f}(x)$, $j=5$'}, savemode=True,
             filepath=r'C:\Users\Máté\Dropbox\CEU\2017 Spring\Nonparametric\Nonparametric_ps1\Problem1_b',
             viewmode=True)

# Part (c) #
# kernel density estimation for balcks and whites, Epa kernel with Silverman's bandwidth, plot
# new domain
support = np.arange(0, max(bw_white.max(), bw_black.max()) + 0.25 * max(bw_white.max(), bw_black.max()), 100)
# estimation
fhat_black_s = [kde.kde_pdf(x=x, sampledata=bw_black, kerneltype='epanechnikov', bandwidth=None, bandwidthscale=None,
                            biascorrected=False, correctboundarybias=False) for x in support]
fhat_white_s = [kde.kde_pdf(x=x, sampledata=bw_white, kerneltype='epanechnikov', bandwidth=None, bandwidthscale=None,
                            biascorrected=False, correctboundarybias=False) for x in support]
# plot
kde.kde_plot(fhat=[fhat_black_s, fhat_white_s], ismultiple=True,
             fsupport=support, plottitle='KDE of black and white birthweight, Epa. with Silverman\'s bandwidth',
             xlabel='gram', ylabel='$\hat{f}(x)$', color={'fhat_1': 'red', 'fhat_2': 'royalblue'},
             legendlabel={'dist_1': 'Black', 'dist_2': 'White'}, savemode=True,
             filepath=r'C:\Users\Máté\Dropbox\CEU\2017 Spring\Nonparametric\Nonparametric_ps1\Problem1_c',
             viewmode=True)
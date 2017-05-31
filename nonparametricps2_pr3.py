# Nonparametric Econometrics Problem set 3 Problem 3 Part (b)-(c)-(d)-(e). Author: Mate Kormos

# Import dependences
import numpy as np
from tabulate import tabulate
from nonparaecon import kde, nptests

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

# Size
n = len(age_b)

# Get Silverman's bandwidth for Gaussian and Epa kernels
# gaussian
h_gau_weight_b = kde.kde_pdf(x=None, sampledata=weight_b, kerneltype='gaussian', kernelorder=2, getsilverman=True)
h_gau_weight_w = kde.kde_pdf(x=None, sampledata=weight_w, kerneltype='gaussian', kernelorder=2, getsilverman=True)
h_gau_age_b = kde.kde_pdf(x=None, sampledata=age_b, kerneltype='gaussian', kernelorder=2, getsilverman=True)
h_gau_age_w = kde.kde_pdf(x=None, sampledata=age_w, kerneltype='gaussian', kernelorder=2, getsilverman=True)
# epa
h_epa_weight_b = kde.kde_pdf(x=None, sampledata=weight_b, kerneltype='epanechnikov', kernelorder=2, getsilverman=True)
h_epa_weight_w = kde.kde_pdf(x=None, sampledata=weight_w, kerneltype='epanechnikov', kernelorder=2, getsilverman=True)
h_epa_age_b = kde.kde_pdf(x=None, sampledata=age_b, kerneltype='epanechnikov', kernelorder=2, getsilverman=True)
h_epa_age_w = kde.kde_pdf(x=None, sampledata=age_w, kerneltype='epanechnikov', kernelorder=2, getsilverman=True)

# Compute the test statistics for different kernels and scaled bandwidths
# summary to print as table at the end
matrixtoprint = np.empty((3, 4))
matrixindex = 0
subsamplesize = 100
for scale in [1 / 3, 1, 3]:
    print('Working on: blacks, Guassian, scale=', scale)
    tstat_gau_b = nptests.nptests_ahmadli(xdata=age_b, ydata=weight_b, bandwidthx=scale * h_gau_age_b,
                                          bandwidthy=scale * h_gau_weight_b,
                                          kerneltype='gaussian', subsamplesize=subsamplesize)
    print('Working on: blacks, Epa, scale=', scale)
    tstat_epa_b = nptests.nptests_ahmadli(xdata=age_b, ydata=weight_b, bandwidthx=scale * h_epa_age_b,
                                          bandwidthy=scale * h_epa_weight_b,
                                          kerneltype='epanechnikov', subsamplesize=subsamplesize)
    print('Working on: whites, Gaussian, scale=', scale)
    tstat_gau_w = nptests.nptests_ahmadli(xdata=age_w, ydata=weight_w, bandwidthx=scale * h_gau_age_w,
                                          bandwidthy=scale * h_gau_weight_w,
                                          kerneltype='gaussian', subsamplesize=subsamplesize)
    print('Working on: whites, Epa, scale=', scale)
    tstat_epa_w = nptests.nptests_ahmadli(xdata=age_w, ydata=weight_w, bandwidthx=scale * h_epa_age_w,
                                          bandwidthy=scale * h_epa_weight_w,
                                          kerneltype='epanechnikov', subsamplesize=subsamplesize)
    # fill in the summary
    matrixtoprint[matrixindex, :] = np.array([tstat_gau_b, tstat_epa_b, tstat_gau_w, tstat_epa_w])
    matrixindex = matrixindex + 1
    # print separately
    print('Test-stat, blacks, Gaussian kernel, ', str(scale), 'times Silvermans bandwidth is', tstat_gau_b)
    print('Test-stat, blacks, Epa kernel, ', str(scale), 'times Silvermans bandwidth is', tstat_epa_b)
    print('Test-stat, whites, Gaussian kernel, ', str(scale), 'times Silvermans bandwidth is', tstat_gau_w)
    print('Test-stat, whites, Epa kernel, ', str(scale), 'times Silvermans bandwidth is', tstat_epa_w, '\n')

# print the summary
print('Summary, not winsorized, subsamplesize=', str(subsamplesize), '\n',
      tabulate(matrixtoprint, tablefmt='latex', floatfmt='.4f'))


# Winsorize data and recompute the statistics
print('\n############################################################################################################\
      \n WINSORIZED \n \
      #########################################################################################################\n')
# x limits
age_lowlim_b = np.percentile(age_b, 2)
age_uplim_b = np.percentile(age_b, 98)
age_lowlim_w = np.percentile(age_w, 2)
age_uplim_w = np.percentile(age_w, 98)
# y limits
weight_lowlim_b = np.percentile(weight_b, 2)
weight_uplim_b = np.percentile(weight_b, 98)
weight_lowlim_w = np.percentile(weight_w, 2)
weight_uplim_w = np.percentile(weight_w, 98)

# keep observations
age_wins_b = list()
weight_wins_b = list()
age_wins_w = list()
weight_wins_w = list()
# blacks
for i in range(n):
    if not(age_b[i] < age_lowlim_b or age_b[i] > age_uplim_b or weight_b[i] < weight_lowlim_b \
    or weight_b[i] > weight_uplim_b):
        age_wins_b.append(age_b[i])
        weight_wins_b.append(weight_b[i])
# whites
for i in range(n):
    if not(age_w[i] < age_lowlim_w or age_w[i] > age_uplim_w or weight_w[i] < weight_lowlim_w \
    or weight_w[i] > weight_uplim_w):
        age_wins_w.append(age_w[i])
        weight_wins_w.append(weight_w[i])

# convert to np array
age_wins_b = np.array(age_wins_b)
weight_wins_b = np.array(weight_wins_b)
age_wins_w = np.array(age_wins_w)
weight_wins_w = np.array(weight_wins_w)

# get bandwidths
# gaussian
h_gau_weight_b = kde.kde_pdf(x=None, sampledata=weight_wins_b, kerneltype='gaussian', kernelorder=2, getsilverman=True)
h_gau_weight_w = kde.kde_pdf(x=None, sampledata=weight_wins_w, kerneltype='gaussian', kernelorder=2, getsilverman=True)
h_gau_age_b = kde.kde_pdf(x=None, sampledata=age_wins_b, kerneltype='gaussian', kernelorder=2, getsilverman=True)
h_gau_age_w = kde.kde_pdf(x=None, sampledata=age_wins_w, kerneltype='gaussian', kernelorder=2, getsilverman=True)
# epa
h_epa_weight_b = kde.kde_pdf(x=None, sampledata=weight_wins_b, kerneltype='epanechnikov', kernelorder=2,
                             getsilverman=True)
h_epa_weight_w = kde.kde_pdf(x=None, sampledata=weight_wins_w, kerneltype='epanechnikov', kernelorder=2,
                             getsilverman=True)
h_epa_age_b = kde.kde_pdf(x=None, sampledata=age_wins_b, kerneltype='epanechnikov', kernelorder=2, getsilverman=True)
h_epa_age_w = kde.kde_pdf(x=None, sampledata=age_wins_w, kerneltype='epanechnikov', kernelorder=2, getsilverman=True)


# repeat test
# summary to print as table at the end
matrixtoprint = np.empty((3, 4))
matrixindex = 0
for scale in [1 / 3, 1, 3]:
    print('Working on: blacks, Guassian, scale=', scale)
    tstat_gau_b = nptests.nptests_ahmadli(xdata=age_wins_b, ydata=weight_wins_b, bandwidthx=scale * h_gau_age_b,
                                          bandwidthy=scale * h_gau_weight_b,
                                          kerneltype='gaussian', subsamplesize=subsamplesize)
    print('Working on: blacks, Epa, scale=', scale)
    tstat_epa_b = nptests.nptests_ahmadli(xdata=age_wins_b, ydata=weight_wins_b, bandwidthx=scale * h_epa_age_b,
                                          bandwidthy=scale * h_epa_weight_b,
                                          kerneltype='epanechnikov', subsamplesize=subsamplesize)
    print('Working on: whites, Guassian, scale=', scale)
    tstat_gau_w = nptests.nptests_ahmadli(xdata=age_wins_w, ydata=weight_wins_w, bandwidthx=scale * h_gau_age_w,
                                          bandwidthy=scale * h_gau_weight_w,
                                          kerneltype='gaussian', subsamplesize=subsamplesize)
    print('Working on: whites, Epa, scale=', scale)
    tstat_epa_w = nptests.nptests_ahmadli(xdata=age_wins_w, ydata=weight_wins_w, bandwidthx=scale * h_epa_age_w,
                                          bandwidthy=scale * h_epa_weight_w,
                                          kerneltype='epanechnikov', subsamplesize=subsamplesize)
    # fill in the summary
    matrixtoprint[matrixindex, :] = np.array([tstat_gau_b, tstat_epa_b, tstat_gau_w, tstat_epa_w])
    matrixindex = matrixindex + 1
    # print separately
    print('Test-stat, blacks, Gaussian kernel, ', str(scale), 'times Silvermans bandwidth is', tstat_gau_b)
    print('Test-stat, blacks, Epa kernel, ', str(scale), 'times Silvermans bandwidth is', tstat_epa_b)
    print('Test-stat, whites, Gaussian kernel, ', str(scale), 'times Silvermans bandwidth is', tstat_gau_w)
    print('Test-stat, whites, Epa kernel, ', str(scale), 'times Silvermans bandwidth is', tstat_epa_w, '\n')

# print the summary
print('Summary, winsorized, subsamplesize=', str(n), '\n',
      tabulate(matrixtoprint, tablefmt='latex', floatfmt='.4f'))
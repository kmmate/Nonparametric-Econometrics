# This file demonstrates the use of functions on npregresssion.py: Nadaraya-Watson estimator,
#  local polynomial regression and plotting tools

# Import dependencies
import numpy as np
from nonparaecon import npregression

#### GENERATE DATA #####

# generate x
# scalar
x = np.arange(-5, 10, 0.04)
# multi
x1, x2 = np.random.uniform(low=-5, high=5, size=100), np.random.uniform(low=-5, high=10, size=100)
X = np.matrix([x1, x2]).T
# generate y
# data generating processes as functions


def datagenproc(x):
    y = 20 + x ** 2 * np.cos(0.5 * x) * int(x < 0) + 10 * np.sin(x) * np.cos(x) * x * int(0 <= x)
    return y


def datagenproc2(x1, x2):
    y = x1 * x2
    return y


# generate y and add noise
# scalar x
y = [datagenproc(x=i) for i in x]
ynoise = [datagenproc(x=i) + 20 * np.random.randn(1)[0] for i in x]
# multiple x
y1 = [datagenproc2(i[0, 0], i[0, 1]) for i in X]
ynoise1 = [datagenproc2(i[0, 0], i[0, 1]) + 10 * np.random.randn(1)[0] for i in X]

#### ESTIMATION ####

# Call the Nadaraya-Watson estimator and the local polynomial regression function, obtain standard errors for the latter
# scalar x
# Nadaraya-Watson
yhat_nw = [npregression.npregression_nadwats(x=i, xdata=x, ydata=ynoise, kerneltype='triangular', bandwidth=0.6)
           for i in x]
# local polynomial regression
polorder = 3
# do cross validation to get the optimal bandwidth
h_opt, sse = npregression.npregression_locpollscv(searchrange=np.arange(2, 5, 0.5), xdata=x,
                                                         ydata=ynoise,
                                                         polorder=polorder, kerneltype='triangular', subsamplesize=None,
                                                         get_SSElist=True)
print('\nh_opt=\n', h_opt, '\nhlist, sse=\n', sse)
# pre-allocation
yhat, se, ci_low, ci_high = list(), list(), list(), list()
for i in x:
    # compute mhat, SE, and CI for each data point
    actual_yhat, actual_se, actual_ci_low, actual_ci_high = npregression.npregression_locpol(x=i, xdata=x,
                                                                                             ydata=ynoise,
                                                                                             polorder=polorder,
                                                                                             kerneltype='triangular',
                                                                                             bandwidth=h_opt,
                                                                                             getSE=True,
                                                                                             getCI=True)
    # append to list
    yhat.append(actual_yhat)
    se.append(actual_se)
    ci_low.append(actual_ci_low)
    ci_high.append(actual_ci_high)
# multiple x
# Nadaraya-Watson
yhat_nw1 = [npregression.npregression_nadwats(x=np.array([i[0, 0], i[0, 1]]), xdata=np.array(X), ydata=ynoise1,
                                              kerneltype='epanechnikov', bandwidth=0.6) for i in X]
# local polynomial regression
# do cross validation to get the optimal bandwidth
h_opt1, sse1 = npregression.npregression_locpollscv(searchrange=np.arange(4, 10, 0.5), xdata=X,
                                                            ydata=ynoise1, polorder=polorder - 1,
                                                            kerneltype='triangular', subsamplesize=200,
                                                            get_SSElist=True)
print('\nh_opt1=\n', h_opt1, '\nhlist1, sse1=\n', sse1)
# regression
yhat1 = [npregression.npregression_locpol(x=np.matrix([i[0, 0], i[0, 1]]), xdata=X, ydata=ynoise1,
                                          polorder=polorder - 1, kerneltype='gaussian', bandwidth=h_opt1,
                                          getCI=False, getSE=False) for i in X]


#### VISUALISATION ####

# Compare Nad-Wats and loc. pol. reg to each other
# scalar x
npregression.npregression_plot(mhat=[yhat_nw, yhat], ismultiple=True, xdomain=x,
                               plottitle='Nad-Wat estimates and local polynomial regression, scalar $x$, $p=$'
                                         + str(polorder),
                               xlabel='$x$', ylabel='$y, \hat{m}(x)$', mtrueon=True, mtrue=[y, y], truestyle='.',
                               confidinton=[False, True], confidint=dict({'ci_low_2': ci_low, 'ci_high_2': ci_high}),
                               legendlabel=dict({'y_1': 'Nad.-Wats.', 'y_2': 'Loc. Pol.'}),
                               seriesperrow=2, savemode=False,
                               filepath=r'C:\Users\Máté\Dropbox\CEU\2017 Spring\Nonparametric\proba',
                               viewmode=True)
# multiple
npregression.npregression_plot(mhat=[yhat_nw1, yhat1], ismultiple=True, xdomain=[x1, x2],
                               plottitle='Nad.-Wats. vs loc. pol. reg., d-vector $x$, $p=$' + str(polorder),
                               xlabel='$x_1$', ylabel='$x_2$', mtrueon=False, mtrue=y1, confidinton=True,
                               subplottitle=dict({'y_1': 'Nad.-Wats. ', 'y_2': 'Loc. Pol. Reg.'}),
                               savemode=False, filepath=r'C:\Users\Máté\Dropbox\CEU\2017 Spring\Nonparametric\proba',
                               viewmode=False)
# Compare to the true y values
npregression.npregression_plot(mhat=[yhat_nw1, yhat1], ismultiple=True, xdomain=[x1, x2],
                               plottitle='Nad.-Wats. vs loc. pol. reg., d-vector $x$, $p=$' + str(polorder),
                               xlabel='$x_1$', ylabel='$x_2$', mtrueon=True, mtrue=[y1, y1], confidinton=True,
                               subplottitle=dict({'y_1': 'Nad.-Wats. ', 'y_2': 'Loc. Pol. Reg.'}),
                               savemode=False, filepath=r'C:\Users\Máté\Dropbox\CEU\2017 Spring\Nonparametric\proba',
                               viewmode=True)
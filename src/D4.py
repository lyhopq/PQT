#%%
import numpy as np 
import scipy.stats as stats
import scipy.optimize as opt


#%%
rv_unif = stats.uniform.rvs(size=10)
print('uniform:\n', rv_unif)
rv_beta = stats.beta.rvs(size=10, a=4, b=2)
print('beta:\n', rv_beta)

#%%
np.random.seed(seed=2015)
rv_beta = stats.beta.rvs(size=10, a=4, b=2)
print('method 1:\n', rv_beta)

np.random.seed(seed=2015)
beta = stats.beta(a=4, b=2)
print('method 2:\n', beta.rvs(size=10))

#%%
norm_dist = stats.norm(loc=.5, scale=2)
data = norm_dist.rvs(size=200)
print('mean of data is:', np.mean(data))
print('median of data is:', np.median(data))
print('standard deviation of data is:', np.std(data))

mu = np.mean(data)
sigma = np.std(data)
stat_val, p_val = stats.kstest(data, 'norm', (mu, sigma))
print('KS-statistic D={:6.3f} p-value={:6.4f}'.format(stat_val, p_val))

#%%
norm_dist2 = stats.norm(loc=-0.2, scale=1.2)
dat2 = norm_dist2.rvs(size=100)
stat_val, p_val	= stats.ttest_ind(data, dat2, equal_var=False)
print('Two-sample t-statistic D	= %6.3f, p-value = %6.4f'.format(stat_val,	p_val))

#%%
g_dist = stats.gamma(a=2)
print('quantiles of 2, 4 and 5:\n', g_dist.cdf([2, 4, 5]))
print ('Values of 25%, 50% and 90%:\n', g_dist.pdf([0.25, 0.5, 0.95]))

#%%
norm_dist =	stats.norm(loc=0, scale=1.8)
data = norm_dist.rvs(size=100)
info = stats.describe(data)
print('Data Size is:', info[0])
print('Minimum value is:', info[1][0])
print('Maximum value is:', info[1][1])
print('Arithmetic mean is:', info[2])
print('Unbiased bariance is:', info[3])
print('Biased skewness is:', info[4])
print('Biased kurtosis is:', info[5])

#%%
norm_dist = stats.norm(loc=0, scale=1.8)
data = norm_dist.rvs(size=100)
mu, sigma = stats.norm.fit(data)
print('MLE of data mean:\n', mu)
print('MLE of data standard deviation:\n', sigma)

#%%
norm_dist = stats.norm()
data1 = norm_dist.rvs(size=100)
exp_dist = stats.expon()
data2 = exp_dist.rvs(size=100)
cor, pval = stats.pearsonr(data1, data2)
print('Pearson correlation coefficient:', cor)
cor, pval = stats.spearmanr(data1, data2)
print("Spearman's correlation coefficient:", cor)

#%%
x = stats.chi2.rvs(3, size=50)
y = 2.5 + 1.2*x + stats.norm.rvs(size=50, loc=0, scale=1.5)
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
print('Slope of fitted model is:', slope)
print('Intercept of fitted model is:', intercept)
print('R-squared:', r_value**2)

#%%
def rosen(x):
    '''The Rosenbrock function'''
    return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)

x_0 = np.array([0.5,1.6,1.1,0.8,1.2,2])
res = opt.minimize(rosen, x_0, method='nelder-mead', options={'xtol': 1e-8, 'disp': True})
print('Result of minimizing Rosenbrock function via Nelder-Mead Simplex a lgorithm:\n', res)

def rosen_der(x):
    xm = x[1:-1]
    xm_m1 = x[:-2]
    xm_p1 = x[2:]
    der = np.zeros_like(x)
    der[1:-1] = 200*(xm-xm_m1**2) - 400*(xm_p1 - xm**2)*xm - 2*(1-xm)
    der[0] = -400*x[0]*(x[1]-x[0]**2) - 2*(1-x[0])
    der[-1] = 200*(x[-1]-x[-2]**2)
    return der

res = opt.minimize(rosen, x_0, method='BFGS',
                   jac=rosen_der, options={'disp': True})
print('Result of minimizing Rosenbrock function via Broyden-Fletcher-Goldfarb-Shanno algorithm:', res)

#%%
#公式看不懂，先跳过
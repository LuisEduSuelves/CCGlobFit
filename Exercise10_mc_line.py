import corner
import emcee
import numpy as np
import matplotlib.pyplot as p
import scipy.optimize as op
import inspect
import struct

# Choose the "true" parameters.
m_true = -0.9594
b_true = 4.294
f_true = 0.534

# Generate some synthetic data from the model.
N = 50
x = np.sort(10*np.random.rand(N))
yerr = 0.1+0.5*np.random.rand(N)
y = m_true*x+b_true
#p.scatter(x, y, s=8, c='g')
#p.plot(x, y, linestyle='-', c='g', label='y points Before')

# Applies 1st set of error
y += np.abs(f_true*y) * np.random.randn(N)
#p.scatter(x, y, s=8, c='r')
#p.plot(x, y, linestyle='--', c='r', label='y points Middle')

# Applies 2nd set of error
y += yerr * np.random.randn(N)
#p.scatter(x, y, s=8, c='b')
#p.plot(x, y, linestyle='--', c='b', label='y points')
#p.scatter(x, yerr, s=10, c='r')
#p.plot(x, yerr, linestyle='--', c='r', label='err')




# np.ones_like(x) is an arrays of all 1. in the same shape and type of the given array (x)
A = np.vstack((np.ones_like(x), x)).T
""" print('A is inside:\n {}'.format(A))"""

C = np.diag(yerr * yerr)
""" print('C is inside:\n {}'.format(C))"""

#np.linalg implements basic linear algebra
#hunde = np.linalg.solve(C, A)
cov = np.linalg.inv(np.dot(A.T, np.linalg.solve(C, A)))
"""print('np.linalg.solve(C, A) is inside:\n {}'.format(np.linalg.solve(C, A)))"""
#print(cov.shape,A.T.shape,hunde.shape,C.shape,A.shape)

b_ls, m_ls = np.dot(cov, np.dot(A.T, np.linalg.solve(C, y)))


print('b_ls is inside: {}'.format(b_ls))
print('m_ls is inside: {}'.format(m_ls))
y_ls = m_ls*x + b_ls
#p.scatter(x, y_ls, s=8, c='g')
#p.plot(x, y_ls, linestyle='-', c='g', label='y_ls')
#p.legend()
#p.show()


# We are using lnf intesad of f. For now, it should at least be clear that this isnt a bad idea because it will force f to be always positive. 

def lnlike(theta, x, y, yerr):
   m, b, lnf = theta
   model = m*x + b
   inv_sigma2 = 1.0/(yerr**2 + model**2*np.exp(2*lnf))
   return -0.5*(np.sum((y-model)**2*inv_sigma2-np.log(inv_sigma2)))



#nll = lambda *argv: -lnlike(*argv)  
def nll(*argv):
    return -lnlike(*argv)

result = op.minimize(nll, [m_true, b_true, np.log(f_true)], args=(x, y, yerr))
#print("Model is storagin:\n zs = {}, {},\n vs = {}, {},\n errs = {} {}\n".format(x, len(x), y, len(y), yerr, len(yerr)))
nardo = m_true, b_true, np.log(f_true)

print(nll(nardo,x, y, yerr))


m_ml, b_ml, lnf_ml = result["x"]
print('Fitted values of the parameters: {}'.format(result["x"]))
#print(dir(result))
#
# Prior wanted values of the parameters (if not wanted returns 0.0)
def lnprior(theta):
    m, b, lnf = theta
    if -5.0 < m < 0.5 and 0.0 < b < 10.0 and -10.0 < lnf < 1.0:
        return 0.0
    return -np.inf

# Full loh probability function
def lnprob(theta, x, y, yerr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, yerr)

ndim = 3
nwalkers = 100
pos = [result["x"] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
#print(pos)

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y, yerr))
sampler.run_mcmc(pos, 500)

#print(pos)

samples = sampler.chain[:, 50:, :].reshape((-1, ndim))
walker_samples = sampler.chain[:, :, :]
#print('samples {}, of shape {}'.format(samples,samples.shape))
#print('samples {}, of shape {}'.format(samples[:],samples.shape))
 
#p.figure(1)
#p.plot(samples[:,0],label='walker dim = 0', ls = "--")
#p.plot(samples[:,0], ls = "--", color = "black")
for i in range (0,5-1):
    p.figure(i)
    p.plot(walker_samples[i,:,0], ls = "--", linewidth = 0.5)
    meanwalk = np.mean(walker_samples[i,:,0])
    p.hlines(meanwalk,0 ,500, label="{}".format(round(meanwalk,3)))
    p.legend()

  
#p.show()
#p.plot(walker_samples[i,:,0], ls = "--", color = "black", linewidth = 0.2)
#p.hlines(m_true,0,500)

p.ylabel('m')
p.xlabel('steps')



#fig = corner.corner(samples, labels=["$m$", "$b$", "$\ln\,f$"],
#                      truths=[m_true, b_true, np.log(f_true)])
#fig.savefig("triangle.png")




xl = np.array([0, 10])
#p.figure(2)
for m, b, lnf in samples[np.random.randint(len(samples), size=100)]:
    p.plot(xl, m*xl+b, color="k", alpha=0.1)
p.plot(xl, m_true*xl+b_true, color="r", lw=2, alpha=0.8)
p.errorbar(x, y, yerr=yerr, fmt=".k")
#p.legend()



samples[:, 2] = np.exp(samples[:, 2]) # because in samples [:,2] we have lnf (ln of f)
m_mcmc, b_mcmc, f_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(samples, [16, 50, 84],
                                                axis=0)))

print(' m = {}   +^{}-_{}\n n = {}   +{}-{}\n f = {}   +{}-{}\n'.format(m_mcmc[0], m_mcmc[1], m_mcmc[2], b_mcmc[0], b_mcmc[1], b_mcmc[2], f_mcmc[0], f_mcmc[1], f_mcmc[2]))

#p.show()

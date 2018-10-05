# Script that contains methods that can be applied to the different models: such as evaluation of gaussians, likelihood calculations, etc.

import ccglobfit.ccplot
import ccglobfit.data
import ccglobfit.dataset
import ccglobfit.gaussianmodel
import ccglobfit.narrowmodel


import corner
import glob
import emcee
import gzip
import inspect
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import scipy.optimize



def fit_optimize(dataset, model):
	ndim = len(model.get_params())
	print(ndim)
	initial_params = model.get_params()

	def negative_ll(params):
		model.set_params(params)
		if model.is_good():
			output = model.likelihood(dataset, model.alpha)
		else:
			output = np.inf
		#print(output)
		return output
	
	res = scipy.optimize.minimize(negative_ll, initial_params)
	#print(res.x,res.x.shape)
	model.set_params(res.x)
	final_params = model.get_params()
	#print("\n\nfinal params = {}".format(final_params))
	return -negative_ll(final_params)




def fit_mcmc(dataset, model, nwalkers, nsteps):
	ndim = len(model.get_params())
	print(ndim)
	initial_params = model.get_params()

	def positive_ll(params):
		model.set_params(params)
		if model.is_good():
			output = -model.likelihood(dataset, model.alpha)
		else:
			output = -np.inf
		#print(output)
		return output
	
	pos = [initial_params + 1e-3*np.random.randn(ndim) for i in range(nwalkers)]
	sampler = emcee.EnsembleSampler(nwalkers, ndim, positive_ll)
	sampler.run_mcmc(pos, nsteps)

	samples = sampler.chain[:, 200:, :].reshape((-1, ndim))
	print(len(samples))
	print(samples.shape)
	#samples_zwei = sampler.chain[:, :10000, :].reshape((-1, ndim))
	#print(samples)
	#fig = corner.corner(samples)100000
	#fig.tight_layout()
	#fig.savefig("triangle.png")

	final_params = list(map(lambda v: (v[0]), zip(*np.percentile(samples, [50], axis=0))))
	#final_params = model.get_params()
	model.set_params(final_params)
	#print("{}".format(final_params))
	#plt.plot(samples[:,0],ls="-", label='a1')
	#plt.legend()
	#plt.show()
	#plt.plot(samples[:,1],ls="-", label='m1')
	#plt.legend()
	#plt.show()
	#plt.plot(samples[:,2],ls="-", label='s1')
	#plt.legend()
	#plt.show()
	#plt.plot(samples[:,-2],ls="-", label='ampli')	
	#plt.legend()
	#plt.show()
	#plt.plot(samples[:,-1],ls="-", label='alpha')
	#plt.legend()
	#plt.show()

	#print(samples)
	fig = corner.corner(samples[:,30:], labels=["amplitud","alpha"], quantiles=[0.5])
	fig2=[]
	#for i in range(6,8):
	fig2 = corner.corner(samples[:,17:23], labels=["a1","m1","s1","a2","m2","s2"], quantiles=[0.5])
		#fig3 = corner.corner(samples[:,3:6], labels=["a","m","s"])
	
	#fig.tight_layout()
	#fig.savefig("Mcmc_model.png")
	return positive_ll(final_params)









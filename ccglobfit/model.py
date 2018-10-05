import ccglobfit.ccplot
import ccglobfit.data
import ccglobfit.dataset
import ccglobfit.gauss
import ccglobfit.gaussianmodel
import ccglobfit.methods

import corner
import emcee
import gzip
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import scipy.optimize

# no idea of what is this
import logging
logger = logging.getLogger(__name__)


class Model():
	def __init__(self, models, amplitude, alpha, color="red", name="default models"):
		
		self.nwmodels = models
		self.amplitude = amplitude
		self.alpha = alpha
		weight = np.array([0.95, 1.44, 2.21, 1.62, 1.37])
		self.weight = weight

	def __str__(self):
		return "amplitude = {}".format(self.amplitude)

	def get_params(self):
		"""
		Gives an array of 5*n*3 numbers
		"""		
		params_arrays = []
		for i in range(0,len(self.nwmodels)):
			params_arrays += self.nwmodels[i].get_params()
		params_arrays += [self.amplitude,self.alpha]
		return params_arrays


	def set_params(self, param_array):
		"""" Takes an array of 16 numbers, and remakes the Model()
		"""
		length = []
		summ = 0
		length.append(0)
		#length.append(len(self.nwmodels[0].gaussians))
		for i in range(0,len(self.nwmodels)):
			summ += self.nwmodels[i].length()
			length.append(summ)


		for j in range(0,len(self.nwmodels)):
			self.nwmodels[j].set_params(param_array[length[j]:length[j+1]])

		self.amplitude = param_array[-2]
		self.alpha = param_array[-1]
		"""self.alpha = param_array[-1] as param_array[len(param_array)-1] is the same
		"""


	def is_good(self):
		"""Returns True if the model parameters are allowed, False otherwise
		"""
		output = True
		for nw in self.nwmodels:
			#print(nw)
			if not nw.is_good():
				output = False		
		return output


	def evaluate(self, alpha=0.0, dataset=None, xs=None):
		sum_weight = np.sum(self.weight)
		ev_array = []
		narrow_sum = 0
		if dataset is None and xs is None:
			raise ValueError("Provide either dataset or xs.")
		if (xs is None):
			for i in range(0,len(self.nwmodels)):
				interm = self.nwmodels[i].evaluate(dataset.narrowdata[i].zs,alpha)
				ev_array.append(interm)
				narrow_sum += interm*self.weight[i] 

		else:
			for i in range(0,len(self.nwmodels)):
				interm = self.nwmodels[i].evaluate(xs,alpha)
				ev_array.append(interm)
				narrow_sum += interm*self.weight[i] 		

		ev_array.append(narrow_sum*self.amplitude/sum_weight)
		return ev_array

	
	def likelihood(self, dataset, alpha=0.0):
		nw_nll = 0
		ev_array = self.evaluate(self.alpha, dataset=dataset)

		for i in range(0,len(self.nwmodels)):
			interm = np.sum(0.5*((dataset.narrowdata[i].vs - ev_array[i])/dataset.narrowdata[i].errs)**2 + 0.5*np.log(2*math.pi*dataset.narrowdata[i].errs**2))
			nw_nll += interm

		broad = np.sum(0.5*((dataset.broaddata.vs - ev_array[-1])/dataset.broaddata.errs)**2 + 0.5*np.log(2*math.pi*dataset.broaddata.errs**2))

		total_nll = broad + nw_nll
		return total_nll
	




	def plot(self, dataset, ccplot):
		""" After a likelihood calculation, with the dataset % parameters for individual bins
		makes the plot of both individuals and the broad bin
		"""		
		xmin=0.0
		xmax=2,
		xs = np.linspace(xmin, xmax, 1000)
		model_lines = self.evaluate(self.alpha, xs=xs)
		ampi = self.amplitude
		ccplot.ax6.plot(xs,model_lines[5], label="A = {0:.3f}\nalpha = {1:.3f}".format(self.amplitude,self.alpha))
		ccplot.ax6.legend(loc = "upper right")
		sum_weight = np.sum(self.weight)
		ccplot.ax6.plot(xs,model_lines[0]*self.weight[0]*self.amplitude/sum_weight, lw=0.75)
		ccplot.ax6.plot(xs,model_lines[1]*self.weight[1]*self.amplitude/sum_weight, lw=0.75)
		ccplot.ax6.plot(xs,model_lines[2]*self.weight[2]*self.amplitude/sum_weight, lw=0.75)
		#ccplot.ax6.plot(xs,model_lines[3],label="original",color="silver", lw=0.5)
		ccplot.ax6.plot(xs,model_lines[3]*self.weight[3]*self.amplitude/sum_weight, color="black", lw=0.75)
		ccplot.ax6.plot(xs,model_lines[4]*self.weight[4]*self.amplitude/sum_weight, lw=0.75)


		plot_gaussians = True
		self.nwmodels[0].plot(dataset.narrowdata[0], self.alpha, ccplot.ax1, plot_gaussians)
		#texstr = "a = {0:.4f}\nm = {1:.4f}\ns ={2:.4f}".format(self.nwmdl[0].a, self.nwmdl[0].m, self.nwmdl[0].s)
		#ccplot.ax1.text(0.055, -0.0085, texstr, fontsize=10)

		self.nwmodels[1].plot(dataset.narrowdata[1], self.alpha, ccplot.ax2, plot_gaussians)

		self.nwmodels[2].plot(dataset.narrowdata[2], self.alpha, ccplot.ax3, plot_gaussians)

		self.nwmodels[3].plot(dataset.narrowdata[3], self.alpha, ccplot.ax4, plot_gaussians)

		self.nwmodels[4].plot(dataset.narrowdata[4], self.alpha, ccplot.ax5, plot_gaussians)


	def write_mdl(self, filepath, protocol = -1):
		if os.path.splitext(filepath)[1] == ".gz":
			pkl_file = gzip.open(filepath, 'wb')
		else:
			pkl_file = open(filepath, 'wb')
		pickle.dump(self, pkl_file, protocol)
		pkl_file.close()
		logger.info("Wrote %s" % filepath)
	
		
	def read_mdl(self, filepath):
		if os.path.splitext(filepath)[1] == ".gz":
			pkl_file = gzip.open(filepath,'rb')
		else:
			pkl_file = open(filepath, 'rb')
		self = pickle.load(pkl_file)
		pkl_file.close()
		logger.info("Read %s" % filepath)
		return self



	def length(self):
		return len(self.get_params())






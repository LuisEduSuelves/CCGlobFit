import ccglobfit.ccplot
import ccglobfit.data
import ccglobfit.dataset
import ccglobfit.gauss
import ccglobfit.gaussianmodel
import ccglobfit.methods

import corner
import emcee
import os
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.optimize



class NarrowModel():
	def __init__(self, models, color="red", name="default model"):
		"""
		Initialize self.gaussian with an array of n(=len(models))*GaussianModel()
		"""
		self.gaussians = models
		self.color = color

	def __str__(self):
		printstr = []
		for i in range(0,len(self.gaussians)): 
			printstr.append("{}".format(self.gaussians[i]))
		return "Innerly = {}".format(printstr)


	def get_params(self):
		"""
		Gives an array of n*3 numbers, where n is the number of gaussians
		"""
		params_arrays = []
		for i in range(0,len(self.gaussians)):
			params_arrays += self.gaussians[i].get_params()
		return params_arrays

	def set_params(self, param_array):
		""" 
		Takes an array of n*3 numbers, where n is the number of gaussians, and remakes the NarrowModel()
		"""
		for i in range(0,len(self.gaussians)):
			self.gaussians[i].set_params(param_array[3*i:3*i+3])

	def is_good(self):
		"""Returns True if the model parameters are allowed, False otherwise
		"""
		output = True

		for g in self.gaussians:
			if not g.is_good():
				output = False
	
		return output

	def evaluate(self, xs, alpha=0.0):
		"""
		Returns the narrowmodel(xs), that is the ys corresponding to the xs
		"""
		return np.sum([g.evaluate(xs) for g in self.gaussians], axis = 0) * (1.0 + xs)**alpha

	def plot(self, data, alpha=0.0, ax=None, plot_gaussians=True):
		xmin=0.0
		xmax=2,
		xs = np.linspace(xmin, xmax, 1000)
		ys = self.evaluate(xs, alpha=alpha)
		#ys = self.evaluate(data.zs, alpha=alpha)
		print(alpha)
		show_plot_afterwards = False
		if ax is None:
			ax = plt.subplot()
			show_plot_afterwards = True

		ax.plot(xs, ys, color="blue")
		if plot_gaussians:
			for g in self.gaussians:
				g.plot(xs, alpha, ax, ls="--")	


		if show_plot_afterwards:
			plt.show()

	def length(self):
		return len(self.get_params())



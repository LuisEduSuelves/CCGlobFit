import ccglobfit.ccplot
import ccglobfit.data
import ccglobfit.dataset
import ccglobfit.gauss
import ccglobfit.methods


import emcee
#import gauss
import os
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.optimize


class GaussianModel():

	def __init__(self, a, m, s, color="red", name="default model"):
		self.a = a
		self.m = m
		self.s = s

		self.color=color
		self.name=name
	
	def get_params(self):
		return self.a, self.m, self.s

	def set_params(self, params):
		"""self.a = params[0]
		self.m = params[1]
		self.s = params[2]
		"""
		self.a, self.m, self.s = params

	def is_good(self):
		"""Returns True if the model parameters are allowed, False otherwise
		"""
		if (self.a < 0) or (self.s < 0):
		#if (self.a < 0) or (self.m > 2) or (self.m < 0) or (self.s < 0.09):
			output = False
		else:
			output = True
		return output

	def evaluate(self, xs, alpha=0.0):
		"""Takes a numpy array x, and returns a corresponding array of y
		"""
		return self.a*np.exp(-((xs-self.m)**2)/(2*self.s**2)) * (1.0 + xs)**alpha

	def __str__(self):
		return "Model Gaussian with a={}, m={}, s={}".format(self.a, self.m, self.s)

	def plot(self, xs_prueba, alpha=0.0, ax=None, xmin=0, xmax=2, **kwargs):
		xs = np.linspace(xmin, xmax, 1000)
		ys = self.evaluate(xs_prueba, alpha)

		show_plot_afterwards = False
		if ax is None:
			ax = plt.subplot()
			show_plot_afterwards = True
		ax.plot(xs_prueba, ys, color=self.color, label=self.name, **kwargs)

	
		if show_plot_afterwards:
			plt.show()
	
	def length(self):
		return len(self.get_params())





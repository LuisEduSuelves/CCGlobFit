# Defines a gaussian distribution given 3 initial parameters: amplitude (A), mean (m) and std. deviation (s)
# includes a Gauss() method, returns the values of a point x or array x[] given the parameters
# includes a Gauss.plot method, returns the plot of a point x or array x[] given the parameters

import matplotlib.pyplot as plt
import numpy as np



class Gauss():
	def __init__(self, A, m, s):
		"""Initializing amplitude (A), mean (m) and std. deviation (s) for a Gaussian 
		distribution"""
		self.A = A
		self.m = m
		self.s = s

	def __call__(self, x):
		return self.A*np.exp(-((x-self.m)**2)/(2*self.s**2))

	def plot(self, x, ax=None):
		show_plot_afterwards = False
		if ax is None:
			ax = plt.subplot()
			show_plot_afterwards = True
		ax.plot(x, self(x),  ls="--")
		if show_plot_afterwards:
			plt.show()
		""" Here is were self() happens, i.e. the object is called -> tha gaussian is calculated
		"""

	def __str__(self):
		return "Gaussian curve with: (A = {self.A}, m = {self.m}, s = {self.s})".format(self=self)



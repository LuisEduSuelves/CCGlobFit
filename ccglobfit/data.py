# Class for defining a Data object, that basically consist of an arrays of the data from the
# galaxy redshifts obtained with the cross-correlation proccess.
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt



class Data(): 
	"""Data(object) would be the defition por python2 or older, in python3 it is implicit"""
	def __init__(self, zs, vs, errs):
		"""Class to hold data
		"""
		self.zs = zs
		self.vs = vs
		self.errs = errs
		
	
	def __str__(self):
		return "Data object of lenght {}".format(len(self.zs))
	
	@classmethod
	def read(cls, filepath): #What is cls?
		"""Method to read a Data object from a CC-file.
		it equals all the terms in the clumn 0 -> : means all the terms in a line, and 0 specifies the column
		"""
		a = np.loadtxt(filepath)
		zs = a[:, 0] 	
		vs = a[:, 1]
		errs = a[:, 2]
		return cls(zs, vs, errs)

	def plot(self, ax=None):
		show_plot_afterwards = False
		if ax is None:
			ax = plt.subplot()
			show_plot_afterwards = True
		ax.errorbar(self.zs, self.vs, yerr=self.errs)
		#ax.errorbar(self.zs, self.vs, yerr=self.errs, fmt='o')
		if show_plot_afterwards:
			plt.show()
	
	def mean_of_zs(self):
		return np.mean(self.zs)
    
    
    
#










import ccglobfit.data
import ccglobfit.dataset
import ccglobfit.gauss
#import ccglobfit.model
#import ccglobfit.narrowmodel
import glob
#import emcee
import inspect
import os
import numpy as np
import matplotlib.pyplot as plt
#import scipy.optimize as op
#import struct

class CCPlot():
	def __init__(self):
		self.fig, ((self.ax1, self.ax2, self.ax3), (self.ax4, self.ax5, self.ax6)) = plt.subplots(2, 3, figsize=(12,8))

		# In case I want to have a fixed picture size
		#self.fig, ((self.ax1, self.ax2, self.ax3), (self.ax4, self.ax5, self.ax6)) = plt.subplots(2, 3,figsize=(15,15))

		self.ax1.set_title('Redshift 0.1 - 0.3')
		self.ax1.set_xlabel('z')

		self.ax2.set_title('Redshift 0.3 - 0.5')
		self.ax2.set_xlabel('z')

		self.ax3.set_title('Redshift 0.5 - 0.7')
		self.ax3.set_xlabel('z')

		self.ax4.set_title('Redshift 0.7 - 0.9')
		self.ax4.set_xlabel('z')

		self.ax5.set_title('Redshift 0.9 - 1.2')
		self.ax5.set_xlabel('z')

		self.ax6.set_title('Redshift Broad bin')
		self.ax6.set_xlabel('z')
		#plt.legend()
		#plt.legend([ax1, ax2, ax3, ax4, ax5, ax6],["NW1", "NW2", "NW3", "NW4", "NW5", "BB"],loc='upper left')
	

	def save_to_file(self, filepath, **kwargs):
		self.fig.savefig(fname=filepath, **kwargs)

	def place_text(self, string, ax, rel_x, rel_y):
		ax.text(string)



import ccglobfit.ccplot
import ccglobfit.data
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt


class DataSet(): 
	"""Data(object) would be the defition por python2 or older, in python3 it is implicit"""
	def __init__(self, datas):
		"""Class to hold narrowdatas and a broaddata.
		narrowdata[1] is the same as boraddata, that is why we delete this line
		"""
		self.narrowdata = datas 
		self.broaddata = datas[1]
		self.narrowdata = np.delete(self.narrowdata, 1)
	
	def __str__(self):
		return "Data object of lenght {}".format(len(self.broaddata.zs))
	
	@classmethod
	def read(cls, filepaths): 
		"""Method to read a Data object from a CC-file.
		it equals all the terms in the clumn 0 -> : means all the
 		terms in a line, and 0 specifies the column
		"""
		datas = []

		for i in range(len(filepaths)):
			datas.append(ccglobfit.data.Data.read(filepaths[i]))
			
		return cls(datas)
			
			
			
	def plot(self, ccplot):

		self.narrowdata[0].plot(ccplot.ax1)
		self.narrowdata[1].plot(ccplot.ax2)
		self.narrowdata[2].plot(ccplot.ax3)
		self.narrowdata[3].plot(ccplot.ax4)
		self.narrowdata[4].plot(ccplot.ax5)
		self.broaddata.plot(ccplot.ax6)




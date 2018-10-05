import ccglobfit.ccplot
import ccglobfit.data
import ccglobfit.dataset
import ccglobfit.gauss
import ccglobfit.gaussianmodel
import ccglobfit.methods
import ccglobfit.model
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

filenames = glob.glob('data/raw_data_v4_merge/v4_merge_100r1000/CC_0*.bref')
filenames.sort()
filepath=[]
for i in range(len(filenames)):
	filepath.append(os.path.join("".join(filenames[i])))
mydataset = ccglobfit.dataset.DataSet.read(filepath)

nwalkers = 66
nsteps = 25000
ngaussians = 2

filepath_mdl = "ccglobfit/CC_model_bias.gz"
filepath_fig = "ccglobfit/CC_model_bias.png"


mybrmodel_list = []
mybrmodel = []
#for j in range (0,5):
#	for i in range (0,ngaussians):
#		mybrmodel_list.append(ccglobfit.gaussianmodel.GaussianModel(0.0005, 0.12 + j/7 +i/(8+j), 0.12))
#		#print(j,i)
#	mybrmodel.append(ccglobfit.narrowmodel.NarrowModel(mybrmodel_list))
#	mybrmodel_list = []


for j in range (0,3):
	for i in range (0,ngaussians):
		mybrmodel_list.append(ccglobfit.gaussianmodel.GaussianModel(0.005 + j/1000, 0.18 + j/10, 0.1))
	mybrmodel.append(ccglobfit.narrowmodel.NarrowModel(mybrmodel_list))
	mybrmodel_list = []
for i in range (0,ngaussians):
	mybrmodel_list.append(ccglobfit.gaussianmodel.GaussianModel(0.009 ,  0.66 + i*0.2, 0.1 + i*0.02))
mybrmodel.append(ccglobfit.narrowmodel.NarrowModel(mybrmodel_list))
mybrmodel_list = []
for i in range (0,ngaussians):
	mybrmodel_list.append(ccglobfit.gaussianmodel.GaussianModel(0.009 , 0.50 + i/2, 0.1))
mybrmodel.append(ccglobfit.narrowmodel.NarrowModel(mybrmodel_list))
mybrmodel_list = []

amplitude = 1.1
alpha_bias = -15


#read_mymodel = ccglobfit.model.Model(mybrmodel, amplitude, alpha_bias)
#mymodel = read_mymodel.read_mdl(filepath_mdl)
mymodel = ccglobfit.model.Model(mybrmodel, amplitude, alpha_bias)

print("model params",mymodel.get_params())
#hola = mymodel.get_params()
#print(hola[0:28:3])
#print(hola[1:29:3])
#print(hola[2:30:3])
#like = ccglobfit.methods.fit_optimize(mydataset, mymodel)
like = ccglobfit.methods.fit_mcmc(mydataset, mymodel, nwalkers, nsteps)
#pannels = []
#pannels = mymodel.evaluate(mydataset, mymodel.alpha)
#print("Obtained evalaution pannels = {}".format(pannels))
print("Goodness of the model = {}".format(mymodel.is_good()))
print("Likelihood obtained = {:.3f}".format(like))
#print("Final parameters = {}".format(mymodel.get_params()))


#print("model params",mymodel.get_params())
hola = mymodel.get_params()
print(hola[0:28:3])
print(hola[1:29:3])
print(hola[2:30:3])


""" Here I have the ploting line
"""
setax = ccglobfit.ccplot.CCPlot()
mydataset.plot(setax)
#mymodel.write_mdl(filepath_mdl)
mymodel.plot(mydataset, setax)
plt.legend()
plt.tight_layout()
#setax.save_to_file(filepath_fig)
#print("Final parameters = {}".format(mymodel.get_params()))
plt.show()




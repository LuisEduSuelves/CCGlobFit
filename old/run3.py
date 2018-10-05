
import ccglobfit.ccplot
import ccglobfit.data
import ccglobfit.dataset
import ccglobfit.gauss
import ccglobfit.gaussianmodel
import ccglobfit.model
import ccglobfit.methods



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

# no idea of what is this
import logging
logger = logging.getLogger(__name__)

filenames = glob.glob('data/raw_data_v4_merge/v4_merge_100r1000/CC_0*.bref')
filenames.sort()
filepath=[]
for i in range(len(filenames)):
	filepath.append(os.path.join("".join(filenames[i])))
#print(filepath)

mydata = ccglobfit.data.Data.read(filepath[0])
mydata1 = ccglobfit.data.Data.read(filepath[1])

mydataset = ccglobfit.dataset.DataSet.read(filepath)

print(mydataset.narrowdata[0])

plt.subplot()
setax = ccglobfit.ccplot.CCPlot()
mydataset.plot(setax)



mymodel_list = []
for i in range (0,3):
	mymodel_list.append(ccglobfit.gaussianmodel.GaussianModel(0.01, 0.25, 0.1))

mymodel_list.append(ccglobfit.gaussianmodel.GaussianModel(0.01, 0.45, 0.1))
mymodel_list.append(ccglobfit.gaussianmodel.GaussianModel(0.01, 0.65, 0.1))

amplitude = 0.1
mymodels = ccglobfit.model.Model(mymodel_list, amplitude)



print("get_params raw = ",mymodels.get_params())
#mymodels.set_params(mymodels.get_params())
#hola = mymodels.get_params()
#print("\nget_params after set_params = {}\n".format(hola))


nwalkers = 1000
nsteps = 1000

#maxlike = mymodels.fit_mcmc(mydataset, nwalkers, nsteps)
#maxlike = mymodels.fit_optimize(mydataset)


#nw_nll = 0
#for i in range (0,5):
#	intermediario = mymodels.nwmdl[i].nll(mydataset.narrowdata[i])
#	nw_nll = nw_nll + intermediario
#	print(nw_nll)
#print("Sum of the narrow likelihoods = {}".format(nw_nll))
#print("Likelihood o the total = {}".format(mymodels.likelihood(mydataset)))


#print("Amplitude of the global fit = {}".format(mymodels.amplitude))
#print("Maximum likelihood obtained for the global fit = {0:.3f}".format(maxlike))


filepath_mdl = "ccglobfit/mymodel_saved.gz"
filepath_fig = "ccglobfit/CC_model.png"



mymodels.write_mdl(filepath_mdl)
mymodels.plot(mydataset, setax)
plt.legend()
plt.tight_layout()
setax.save_to_file(filepath_fig)
plt.show()

#new_mymodels = mymodels.read_mdl(filepath_mdl)
#new_mymodels.plot(mydataset, setax)
#plt.tight_layout()
#setax.save_to_file(filepath_fig)
#plt.show()


#mymodels.write_mdl(filepath_mdl)
#new_mymodels = mymodels.read_mdl(filepath_mdl)
#print("Amplitude of the global fit = {}".format(new_mymodels.amplitude))




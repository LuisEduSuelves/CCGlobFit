# Code made for working with old data.py and model.py, the later before becoming narrowmodel.py

import ccglobfit.data
#import ccglobfit.gauss
import ccglobfit.narrowmodel
import emcee
import inspect
import os
import numpy as np
import matplotlib.pyplot as plt
#import scipy.optimize as op
#import struct


mymodel = ccglobfit.narrowmodel.NarrowModel(0.01, 0.55, 0.1)

#print(mymodel)
#mymodel.plot()


datadir = "data/raw_data_v4_merge/v4_merge_100r1000"
filename = "CC_0.901z1.201.bref"
filepath = os.path.join(datadir, filename)

mydata = ccglobfit.data.Data.read(filepath)
#print(mydata)

print(mymodel.get_params())
#mymodel.montecarlo_chi(mydata, 100, 3, 1000)

mymodel.montecarlo_nll(mydata, 100, 3, 500)

#mymodel.fit_chi(mydata)

#mymodel.fit_nll(mydata)

#print(mydata)
print(mymodel.get_params())



ax = plt.subplot()
mydata.plot(ax)
mymodel.plot(ax)

plt.show()








import ccglobfit.data
import ccglobfit.gauss
import ccglobfit.model
import inspect
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
import struct

#a = np.linspace(0, 2, 100)
#b = np.random.randn(100)
#c = np.ones(100)

#mydata = ccglobfit.data.Data(a, b, c)

#print(mydata)

#print(mydata.mean_of_zs())



datadir = "data/raw_data_v4_merge/v4_merge_100r1000"
filename = "CC_0.101z0.301_bref.cov"

filenames = []
filepath = os.path.join(datadir, filename)

mydata2 = ccglobfit.data.Data.read(filepath)

print(mydata2)

	

ax1 = plt.subplot()
mydata2.plot(ax1)
		
plt.show()

#print("{}").format(mydata2)
#
x = np.linspace(-6.5, 6.5, 200)
#ax2 = plt.subplot()
#gss = ccglobfit.gauss.Gauss(1, 0, 2)
#gssln = gss(x)
#print(gssln,gss, sep ="\n")
#gss.plot(x, ax2)
#
#plt.show()
#
#
#model = ccglobfit.model.Model(mydata2)
#print(model)

#print(model)
#gauss_array = np.array([1, 2, 0.5])
#print(gauss_array)
#print("lnlike inside model gives the probability: {}".format(model.nll(gauss_array)))

def gaussian(mdl, x):
	A, m, s = mdl
	return A*np.exp(-((x-m)**2)/(2*s**2))

def lnlike(mdl, x, y, yerr):
	A, m, s = mdl
	calc = gaussian(mdl, x)
	sigma = yerr**2 + calc**2
	return -(0.5*(y - calc)**2/sigma + 0.5*np.log(2*sigma*3.14))

#def nllout(*argv):
#	return -lnlike(*argv)

#minim1param = lambda x: lnlike(gauss_array, model.zs[:5], model.vs[:5], model.errs[:5])

#print("param {}".format(minim1param(model.zs)))
#plt.plot(model.zs[:5],minim1param(model.zs[:5]), label="model.nll")
#plt.plot(model.zs[:5],lnlike(gauss_array, model.zs[:5], model.vs[:5], model.errs[:5]), label="model.nll")
#plt.legend()
#plt.show()
#minimo = op.minimize(minim1param, x0 = 0.2)

#print('Fitted value of the paramete x: {}'.format(minimo))
#def nllout(*argv):
#	return -lnlike(*argv)

#print(lnlike(gauss_array,model.zs[:5], model.vs[:5], model.errs[:5]), lnlike(gauss_array, model.zs, model.vs, model.errs).shape)

#print(model.nll(gauss_array))

#plt.plot(model.zs[:5],model.nll(gauss_array)[:5], label="model.nll")
#plt.legend()
#plt.show()


#minimo = op.minimize(nllout, [gauss_array])


#minimo = op.minimize(model.nll, [gauss_array], args=(model.zs, model.vs, model.errs))

#print('Fitted values of the parameters: {}'.format(minimo["x"]))

#gss = ccglobfit.gauss.Gauss(minimo)
#gss.plot(x, ax2)


	

#print(lnlike(model, mydata2.zs, mydata2.vs, mydata2.errs))







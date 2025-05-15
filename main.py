import STOM_higgs_tools, numpy as np
import matplotlib.pyplot as plt # Making plots.
from scipy import integrate


#function to create the histogram plot
def function_hist(a, ini, final):

    # 12 bins
    bins = np.linspace(ini, final, 31)
    weightsa = np.ones_like(a)/float(len(a))
    return np.histogram(np.array(a), bins, weights = weightsa)

def exp_pdf(x,A,lamb):
    return A*np.exp(-x/lamb)

vals = np.asarray(STOM_higgs_tools.generate_data()) #the values of the generated data
bin_heights, bin_edges = function_hist(vals,104, 155) #bin heights are the individual values, and the edges characterise the histogram
plt.errorbar((bin_edges[0:-1:]+bin_edges[1::])/2,bin_heights*len(vals),np.sqrt(bin_heights*len(vals)),ls="none") #errorbars

#calculating lambda
lamb=np.sum(vals)/len(vals)

#calculating the prefactor
A=(np.sum(vals[vals<120])+np.sum(vals[vals>127]))/(integrate.quad(exp_pdf, 104, 120,args=(1,lamb))[0]+integrate.quad(exp_pdf, 127, 155,args=(1,lamb))[0])
print(integrate.quad(exp_pdf, 104, 120,args=(1,lamb))[0])

x=np.linspace(104,155,1000)
plt.plot(x,exp_pdf(x,len(vals)/lamb,lamb))




plt.show()

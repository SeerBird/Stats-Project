import numpy as np
from main import bin_heights, vals,exp_pdf,bin_edges
from scipy import optimize
from matplotlib import pyplot as plt

def chisq(paras,x,y):
    A=paras[1]
    lamb=paras[0]
    return np.sum((y-exp_pdf(x,A,lamb))**2/(x))

y_i=bin_heights*len(vals)
x_i=(bin_edges[0:-1:]+bin_edges[1::])/2
print(chisq([30,60000],x_i,y_i))
plt.scatter(x_i,y_i)
plt.show()
F=optimize.minimize(chisq, x0=[30,60000],args=(x_i,y_i))

print(F)


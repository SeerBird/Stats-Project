import STOM_higgs_tools, numpy as np
import matplotlib.pyplot as plt # Making plots.
from scipy import integrate

#function which performs numerical integration
def trapezium(x,f):
    '''
    Numerical integration via trapezium rule
    takes two args: x values, and f(x) values
    Returns numerical value of the integral
    '''
    integral=0
    n=len(x)
    for i in range(n-1):
        integral+=(x[i+1]-x[i])*(f[i+1]+f[i])
    return 0.5*integral

#function to create the histogram plot
def function_hist(a, ini, final):
    '''
    Function to create a histogram
    inputs: array to work with, initial and final values to be binned.
    Returns two arrays, the first is bin heights and the second is the bin edges.
    '''
    # 30 bins
    bins = np.linspace(ini, final, 31)
    weightsa = np.ones_like(a)/float(len(a))
    return np.histogram(np.array(a), bins, weights = weightsa)

#parametrised exponential PDF
def exp_pdf(x,A,lamb):
    '''
    Returns a scaled exponential distribution
    inputs: x values, scaling prefactor, and lambda coefficient
    '''
    return A*np.exp(-x/lamb)

#creating & plotting the histogram of the generated data.
vals = np.asarray(STOM_higgs_tools.generate_data()) #the values of the generated data
bin_heights, bin_edges = function_hist(vals,104, 155) #bin heights are the individual values, and the edges characterise the histogram
plt.errorbar((bin_edges[0:-1:]+bin_edges[1::])/2,bin_heights*len(vals),np.sqrt(bin_heights*len(vals)),
             ls="none",marker='o',markersize=2.5,capsize=1,elinewidth=1,label='Data',color='black')

#creating array for background (i.e excluding the signal)
peak_high=155 # the top border of the exlusion region
peak_low=135 # bottom border of the exclusio region
BG=np.concatenate((vals[vals<peak_low], vals[vals>peak_high])) #BG is just the vals, without the central region
f_bg,x_bg=function_hist(BG,104, 155) #f_bg height of background bins, x_bg bin edges for background
# NOTE since there is lot of scattering in the <120 region, so I obtained the best fit to exclude the entire low energy region of the dataset

#calculating lambda
lamb=np.average(BG)
print(np.average(bin_heights*len(vals)))

#calculating the prefactor
#integral_data=(np.sum(f_bg*len(BG)))
integral_data=trapezium((x_bg[0:-1:]+x_bg[1::])/2,f_bg*len(BG)) # numerical integration on the given data
integral_func=integrate.quad(exp_pdf, 104, peak_low,args=(1,lamb))[0]+integrate.quad(exp_pdf, peak_high, 155,args=(1,lamb))[0] 
A=integral_data/integral_func # the prefactor is obtained by the ratio of the integral of the PDF and the data

#ploting the background distribution
x=np.linspace(104,155,1000)
plt.plot(x,exp_pdf(x,A,lamb), label='B')


#aesthetics
plt.title('Discovery of the Higgs Boson yay')
plt.xlabel(r'$m_{\gamma\gamma}$ (GeV)')
plt.ylabel('Number of entries')
plt.xticks([120,140])
plt.yticks(np.arange(0,2200,200))
plt.legend()


print('lambda: '+str(lamb)+
      '\n A: ' +str(A))
plt.show()

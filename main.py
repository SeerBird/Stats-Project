import STOM_higgs_tools, numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy.stats import chi2 as chi2_dist
import warnings

warnings.filterwarnings("error")


def background_and_signal_model(params: tuple, E: float | np.ndarray):
    """

    :param params: see unpacking below
    :param E: rest energy of the photon(s)
    :return: photon number density at that energy
    """
    A_background = params[0]
    a = params[1]
    A_signal = params[2]
    mu = params[3]
    sigma = params[4]
    background = A_background * np.exp(-E / a)
    signal = A_signal / (np.sqrt(2. * np.pi) * sigma) * np.exp(-np.power((E - mu) / sigma, 2.) / 2)
    return background + signal


def background_model(_A, _a, _E: float | np.ndarray):
    return _A * np.exp(-_E / _a)


def trapezium_integrate(xarray: np.ndarray,
                        func):
    """
    :param xarray: ordered(descending or ascending) array of xvalues at which the function will be evaluated
    :param func: vectorizable function float->float
    :return: trapezium integral of the functions over the given x values
    """
    yarray: np.ndarray = func(xarray)
    return np.sum((yarray[:-1] + yarray[1:]) * (xarray[1:] - xarray[:-1]) / 2, axis=0)


def get_hist(_vals: np.ndarray, _start: float, _end: float, _bins: int):
    """
    :param _vals: array of values to put into bins
    :param _start: low edge of first bin
    :param _end: high edge of last bin
    :param _bins: number of bins
    :return: tuple containing the array of values per bin(bin heights) and the array of bin middle positions
    """
    _bin_heights, _bin_edges = np.histogram(_vals, np.linspace(_start, _end, _bins + 1))
    return _bin_heights, (_bin_edges[1:] + _bin_edges[:-1]) / 2


data = np.asarray(STOM_higgs_tools.generate_data())
plot_start = 104
plot_end = 155
signal_start = 120
signal_end = 130
without_signal = data[(data < signal_start) | (data > signal_end)]
with_signal = data[(data > plot_start) & (data < plot_end)]  # only fit the relevant region? #TODO: is this correct?
nbins = 30
gaussian_assumption_A = 700
gaussian_assumption_mu = 125
gaussian_assumption_sigma = 1.5
# region Task 1&2
# region plot generated data
bin_heights, binx = get_hist(with_signal, plot_start, plot_end, nbins)
plt.errorbar(binx, bin_heights, np.sqrt(bin_heights),
             ls="none", marker='o', markersize=2.5, capsize=1, elinewidth=1, label='Data', color='black')
# endregion
max_lik_a = np.average(without_signal)  # get the second parameter of the background model - a

bin_heights, binx = get_hist(without_signal, plot_start, plot_end, nbins)
integral_without_signal = np.sum((bin_heights[1:] + bin_heights[:-1]) * (binx[1:] - binx[:-1]) / 2)
trial_model = lambda _x: background_model(1, max_lik_a, _x)  # set A to 1 and a to the value we got earlier
integral_trial_model = (trapezium_integrate(np.linspace(plot_start, signal_start), trial_model) +
                        trapezium_integrate(np.linspace(signal_end, plot_end), trial_model))  # exclude the signal
max_lik_A = integral_without_signal / integral_trial_model  # get the first parameter of the background model - A
print(f"Max-likelihood fit: A = {max_lik_A:.4f}, a = {max_lik_a:.4f}")
# region plot max-likelihood fit
x = np.linspace(plot_start, plot_end, 1000)
plt.plot(x, background_model(max_lik_A, max_lik_a, x), label='Maximum likelihood fit')


# endregion
# endregion
# region Task 3
def chi2(_x: np.ndarray, _y: np.ndarray, _ysig, func):
    """
    :param _x: data x values
    :param _y: data y values
    :param _ysig: data y uncertainties
    :param func: function to get the chi^2 value for
    :return: chi^2 value for given data and function
    """
    return np.sum(((_y - func(_x)) / _ysig) ** 2, axis=0)


def scan_param_ranges(A_values, a_values, _binx, _bin_heights, model):
    """
        :param A_values: values of A to check
        :param a_values: values of a to check
        :param _binx: data x values
        :param _bin_heights: data y values
        :param model: the model to fit,  (A_values, a_values, _binx)->(model y values)
        :return: best A value, best a value, the chi^2 value achieved with those A and a values
        """
    param_mesh = np.meshgrid(A_values, a_values, indexing="ij")
    trial_funcs = lambda _x: model(param_mesh[0], param_mesh[1], _x)

    # append new axes to make the arrays broadcastable with the parameter mesh
    chi2s = chi2(_binx[:, None, None], _bin_heights[:, None, None], np.sqrt(_bin_heights)[:, None, None], trial_funcs)
    fit_index = np.unravel_index(np.argmin(chi2s, axis=None), shape=chi2s.shape)
    return param_mesh[0][fit_index], param_mesh[1][fit_index], chi2s[fit_index]


def least_chi2_fit_A_and_a(A_min, A_max, a_min, a_max, _binx, _bin_heights, model):
    """
    :param A_min: minimum A value to scan
    :param A_max: maximum A value to scan
    :param a_min: minimum a value to scan
    :param a_max: maximum a value to scan
    :param _binx: data x values
    :param _bin_heights: data y values
    :param model: the model to fit,  (A_values, a_values, _binx)->(model y values)
    :return: best A value, best a value, the chi^2 value achieved with those A and a values
    """
    _best_A, _best_a, _best_chi2 = ((A_min + A_max) / 2, (a_min + a_max) / 2, 0)
    checknum = 10  # number of values to check per range
    _iter = 5
    for _i in range(_iter):
        _best_A, _best_a, _best_chi2 = scan_param_ranges(np.linspace(A_min, A_max, checknum),
                                                         np.linspace(a_min, a_max, checknum),
                                                         _binx, _bin_heights, model)
        A_min = _best_A - (A_max - A_min) / checknum
        A_max = _best_A + (A_max - A_min) / checknum
        a_min = _best_a - (a_max - a_min) / checknum
        a_max = _best_a + (a_max - a_min) / checknum
    return _best_A, _best_a, _best_chi2


# background-only hypothesis without signal region
heights_without_signal, binx_without_signal = get_hist(with_signal, np.min(with_signal), np.max(with_signal), nbins)
heights_without_signal = heights_without_signal[  # only take bins outside the signal region
    (binx_without_signal < signal_start) | (binx_without_signal > signal_end)]
binx_without_signal = binx_without_signal[(binx_without_signal < signal_start) | (binx_without_signal > signal_end)]
# region remove zero-height binx
binx_without_signal = binx_without_signal[heights_without_signal != 0]
heights_without_signal = heights_without_signal[heights_without_signal != 0]
# endregion
best_chi2_A, best_chi2_a, background_fit_chi2 = least_chi2_fit_A_and_a(50000, 70000, 28, 31,
                                                                       binx_without_signal,
                                                                       heights_without_signal,
                                                                       background_model)
# region plot chi2 fit
x = np.linspace(plot_start, plot_end, 1000)
plt.plot(x, background_model(best_chi2_A, best_chi2_a, x),
         label=r"Least $\chi^2$ background-only fit")


# endregion

# endregion
# region Task 4
# region a
def chi2_PDF(_chi2: np.ndarray | float, dof: int):
    """
    The probability distribution of chi^2 of a fit given its number of degrees of freedom
    :param _chi2: chi^2 value at which the PDF is evaluated
    :param dof: the number of degrees of freedom for your fit
    :return: probability density in chi^2
    """
    result = np.zeros_like(_chi2)
    result[_chi2 >= 10 * dof] = 0
    smallchi2 = _chi2[_chi2 < 10 * dof]
    k_over_2 = float(dof) / 2
    try:
        result[_chi2 < 10 * dof] = smallchi2 ** (k_over_2 - 1) * np.exp(-smallchi2 / 2) / (
                (2 ** k_over_2) * gamma(k_over_2))
    except RuntimeWarning:
        print(
            "Overflow encountered when trying to compute the chi2 PDF, returned zeros instead"
            " - consider making a better fit...")
    return result


def p_from_chi2(_chi2: float | np.ndarray, dof: int):
    """
    :param _chi2: the chi^2 of your fit
    :param dof: the number of degrees of freedom for your fit
    :return: p-value of your fit
    """
    return 1 - chi2_dist.cdf(_chi2,df = dof)


# region examine the background-only chi2 fit without signal region now that we can
p_background_without_signal = p_from_chi2(background_fit_chi2, len(heights_without_signal) - 2)
print(
    f"Background-only chi^2 fit without signal region: A = {best_chi2_A:.4f}, a = {best_chi2_a:.4f},"
    f" p = {p_background_without_signal:.4f}")
# endregion
# region background-only hypothesis with signal region

heights_with_signal, binx_with_signal = get_hist(with_signal, np.min(with_signal), np.max(with_signal), nbins)
# region remove zero-height bins to avoid dealing with division by zero
binx_with_signal = binx_with_signal[heights_with_signal != 0]
heights_with_signal = heights_with_signal[heights_with_signal != 0]
# endregion
best_chi2_A, best_chi2_a, least_chi2 = least_chi2_fit_A_and_a(50000, 70000,
                                                              28, 31,
                                                              binx_with_signal,
                                                              heights_with_signal,
                                                              background_model)
p_background_with_signal = p_from_chi2(least_chi2, len(heights_with_signal) - 2)
print(f"Background-only chi^2 fit with signal region: A = {best_chi2_A:.4f}, a = {best_chi2_a:.4f},"
      f" p = {p_background_with_signal:.2E}")


# endregion
# endregion
# region b
# region gaussian assumption
def gaussian_assumption(A, a, _E):
    return background_and_signal_model((A, a, gaussian_assumption_A, gaussian_assumption_mu,
                                        gaussian_assumption_sigma), _E)


best_chi2_A, best_chi2_a, gaussian_chi2 = least_chi2_fit_A_and_a(50000, 70000,
                                                                 25, 35,
                                                                 binx_with_signal,
                                                                 heights_with_signal,
                                                                 gaussian_assumption)
p_gaussian_assumption = p_from_chi2(gaussian_chi2, len(heights_with_signal) - 2)
# endregion
print(f"Gaussian assumption chi^2 fit: A = {best_chi2_A:.4f}," +
      f" a = {best_chi2_a:.4f}, p = {p_gaussian_assumption:.4f}")
# region plot chi2 gaussian assumption fit
x = np.linspace(plot_start, plot_end, 1000)
plt.plot(x, background_and_signal_model((best_chi2_A, best_chi2_a, gaussian_assumption_A,
                                         gaussian_assumption_mu, gaussian_assumption_sigma), x),
         label=r"Least $\chi^2$ gaussian assumption fit")


# endregion
# endregion
# endregion
# endregion
# region prettify plot
plt.title('Discovery of the Higgs Boson', fontsize=16)
plt.xlabel(r'$m_{\gamma\gamma}$ (GeV)', fontsize=14)
plt.ylabel('Number of entries', fontsize=14)
plt.xticks([120, 140])
plt.minorticks_on()
plt.yticks(np.arange(0, 2200, 200))
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()
# endregion

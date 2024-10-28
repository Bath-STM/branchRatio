#########################################################
# Designed to analyse the toluene branching ratio data
# Processes manipulation time data into DataFrames
# Calculates bR and reaction rates as function of bias and current
# Kept these in different functions despite similar code to keep options minimal and code simpler to read
# Finds the rates & bR with final point, linear fit and double fit methods
# Evaluates uncertainties with a varity of methods (counting, beta, fit)
# Uses adj R^2 GoF to look at bR/rates as function of n experiments.
# Compares bR from different methods via residuals
# Shows beta statistic probabilty density functions
# Options for these (units and normalisation) are commented in/out 
# Fits for n electron process. N.B. must turn stayedDict on/off as required for the different analyses as required
# Fits multiple Guassians to STS data and then creates franken-STS (tFM) from these states
# Integrates to find the amount of charge captured in each state identified in the Gaussian fitting
# Applies weighting to each state to fit the captured charge to reaction rates (beta param fits) found in earlier functions
# 
# Location guide:
# FC -> faulted corner
# tFM -> toluene faulted middle 
# tFC -> toluene faulted corner (franken STS)
# ------------------------------------------------------- 
# Example command `python3 tolueneBRatioAnalysisDevelop.py`
# -------------------------------------------------------
# TODO - More cunning way to deal with stayedDict
#########################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.ticker as ticker
from scipy.optimize import curve_fit
from functools import partial
from sklearn.metrics import r2_score
from scipy.stats import beta
from sklearn.metrics import auc
from math import log10, floor

import sys

def line(x, slope):
    """Line with intercept fixed at 0
    
    Arguments
	---------
	x : np.array([float])
    slope : float
	
    Returns
	-------
	slope*x : pd.Series<float>
        The corresponding y value for a given x value and slope
    """
    return slope*x

def line_c(x, slope, c):
    """Line with intercept
    
    Arguments
	---------
	x : np.array([float])
    slope : float
    c : float

	Returns
	-------
	(slope*x)+c : pd.Series<float>
        The corresponding y value for a given x value, slope and constant
    """
    return (slope*x)+c

def power_law(x, B, n):
    """Power law expression with a scaling constant 
    
    Arguments
	---------
	x : np.array([float])
		The x values
    B : float
        The scaling prefactor
    n : float
        The power
        
	Returns
	-------
	(B*x)**n : pd.Series<float>
        The corresponding y value for a given x value, prefactor and power
    """
    return (B*x)**n

def gaussian(x, *popt):
    """Gaussian distribution
    
    Arguments
	---------
	x : np.array([float])
		The x values
    mu : float
        The mean value/centre of the distribution
    sigma : float
        The width of the distribution
    A : float
        The amplitude of the distribution
        
	Returns
	-------
	(B*x)**n : pd.Series<float>
        The corresponding y value for a given x value, amplitude, mean and width
    """
    assert len(popt) == 3, 'The number of args passed must be exactly 3.\nPass [mu, sigma, A] for the Gaussian dist'
    mu, sigma, A = popt
    return(A*np.exp(-(x-mu)**2/(2.*sigma**2)))

def sum_n_gaussians(x, *args):
    """Sum n gaussian distributions.
    Number to use is found based on the number of initial guess params passed
    
    Arguments
	---------
	x : np.array([float])
		The x values
    args : (float, float, float)
        Variable length tuple containing the initial guesses passed to `gaussian` in the fit. 
        N.B. must be in the order [mu, sigma, A] for each Gaussian dist. Thus len must be multiple of 3.
        
	Returns
	-------
	sumGauss : pd.Series<float>
        The corresponding y value found with n Gaussians for a given set of x values, amplitudes, means and widths
    """
    # check we have the right number of inputs
    assert len(args) % 3 == 0, 'The number of args passed must be a multiple 3.\nPass [mu, sigma, A] for each Gaussian dist'
    # convert back into list of lists. Don't have to do it this way but easier to keep track of which initial vals correspond to each other
    p0List = [args[i:i+3] for i in range(0, len(args), 3)]
    
    sumGauss = sum([gaussian(x, *p0L) for p0L in p0List])
    return sumGauss

def bR_linear(df_masked):
    """Find the branching ratio by fitting line to N mols diffused against N mols desorbed
    
    Arguments
	---------
	df_masked : pd.DataFrame
		DataFrame masked to the desired selction of data 
        
	Returns
	-------
	bR_lin : float
        Branching ratio, k_des/k_diff
    uncert_fit_lin : float
        The uncertainty on the linear fit
    """
    popt, pcov = curve_fit(line, df_masked['nDiff'], df_masked['nDes'])
    bR_lin = popt[0]
    uncert_fit_lin = pcov[0,0]**0.5
    return bR_lin, uncert_fit_lin

def total_rate_fit(t, KTot, N0): # pass N0
    """ Expression for the total manipulation rate
    Function arguments with default value have to be at the end
    Thus, params that we want to pass/fix need to after those we vary
    Designed so N0 can be passed as a known value
    
    Arguments
	---------
	t : np.array([float])
		Usually DataFrame column of the manipulation time masked to the desired selction of data 
    KTot : float
        The total manipulation rate
    N0 : int or float
        The total number of mol manipulated across the full dataset
	Returns
	-------
	N0*(1-np.exp(-1*KTot*t)) : pd.Series<float>
        Number of mols manipulated (in either channel) as a function of time
    """
    return N0*(1-np.exp(-1*KTot*t))

def kx_fit(t, kx, N0, KTot): # pass N0, KTot
    """Expression for the manipulation rate in a given sub channel (des or diff)
    As above, function designed so N0 and KTot can be passed as known values
    
    Arguments
	---------
	t : np.array([float])
		Usually DataFrame column of the manipulation time in the channel masked to the desired selction of data 
    kx : float
        The channel manipulation rate
    N0 : int or float
        The total number of mol manipulated across the full dataset
    KTot : float
        The total manipulation rate
        
	Returns
	-------
	(N0*kx/KTot)*(1-np.exp(-1*KTot*t)) : pd.Series<float>
       Number of mols manipulated in the channel as a function of time
    """
    return (N0*kx/KTot)*(1-np.exp(-1*KTot*t))

def bR_doubleFit(df_masked, N_0):
    """Find the branching ratio by fitting first to the total rate, then to each of the pathways
    
    Arguments
	---------
	df_masked : pd.DataFrame
		DataFrame masked to the desired selction of data 
    N_0 : numpy.int64
        Total number of manipulation events
        
	Returns
	-------
	bR_dbF : float
        Branching ratio, k_des/k_diff
    (uncert_KTot_dbF/KTot, uncert_kdes_dbF/kdes, uncert_kdiff_dbF\kdiff) : (float, float, float)
        Tuple containing the fractional uncertainty on the fit to the total, desorption and diffusion rates
    """
    # First fit total rate with N0 fixed
    total_rate_fit_N0 = partial(total_rate_fit, N0=N_0)
    popt, pcov = curve_fit(total_rate_fit_N0, df_masked['Time'], df_masked['nManip'], p0=[1.0])
    K_Tot = popt[0]
    uncert_KTot_dbF = pcov[0,0]**0.5
    # Now fit k_des and k_diff using N0 and the fitted KTot
    kx_fit_N0_KTot = partial(kx_fit, N0=N_0, KTot=K_Tot)
    popt, pcov = curve_fit(kx_fit_N0_KTot, df_masked['Time'], df_masked['nDes'], p0=[1.0])
    k_des = popt[0]
    uncert_kdes_dbF = pcov[0,0]**0.5
    popt, pcov = curve_fit(kx_fit_N0_KTot, df_masked['Time'], df_masked['nDiff'], p0=[1.0])
    k_diff = popt[0]
    uncert_kdiff_dbF = pcov[0,0]**0.5
    # Divide the two rates to get the BR
    bR_dbF = k_des/k_diff
    return bR_dbF, (uncert_KTot_dbF/K_Tot, uncert_kdes_dbF/k_des, uncert_kdiff_dbF/k_diff)

def path_rates(df_masked, N_0):
    """Find the rates for each of the reaction paths (desorbed and switched)
    
    Arguments
	---------
	df_masked : pd.DataFrame
		DataFrame masked to the desired selction of data 
    N_0 : numpy.int64
        Total number of manipulation events
        
	Returns
	-------
	k_des : float
        The desorbtion rate
    k_diff : float
        The diffusion rate
    K_Tot : float
        The total rate    
    (uncert_kdes_dbF, uncert_kdiff_dbF, uncert_KTot_dbF) : (float, float, float)
        Tuple containing the uncertainty on the fit to the desorption and diffusion rates
    """
    # First fit total rate with N0 fixed
    total_rate_fit_N0 = partial(total_rate_fit, N0=N_0)
    popt, pcov = curve_fit(total_rate_fit_N0, df_masked['Time'], df_masked['nManip'], p0=[1.0])
    K_Tot = popt[0]
    uncert_KTot_dbF = pcov[0,0]**0.5
    # Now fit k_des and k_diff using N0 and the fitted KTot
    kx_fit_N0_KTot = partial(kx_fit, N0=N_0, KTot=K_Tot)
    popt, pcov = curve_fit(kx_fit_N0_KTot, df_masked['Time'], df_masked['nDes'], p0=[1.0])
    k_des = popt[0]
    uncert_kdes_dbF = pcov[0,0]**0.5
    popt, pcov = curve_fit(kx_fit_N0_KTot, df_masked['Time'], df_masked['nDiff'], p0=[1.0])
    k_diff = popt[0]
    uncert_kdiff_dbF = pcov[0,0]**0.5
    return k_des, k_diff, K_Tot, (uncert_kdes_dbF, uncert_kdiff_dbF, uncert_KTot_dbF)

def path_rates_errs(sigmakDiffFit, sigmakDesFit, kDiff, kDes, sigmaBR, bR):
    """Calculate the uncertainty on the rates for each of the reaction paths (desorbed and switched).
    Errors on the individual rates depend on the other rate and the bR
    
    Arguments
	---------
	sigmakDiffFit : [float]
		The uncertainty on the diffusuion rate from the fit (for each bias, current etc.)
    sigmakDesFit : [float]
        The uncertainty on the desorbtion rate from the fit (for each bias, current etc.)
    kDiff: [float]
        The diffusion rate (for each bias, current etc.)
    kDes : [float]
        The desorbtion rate (for each bias, current etc.)
    sigmaBR : [float]
        The uncertainty on the bR (for each bias, current etc.)
    bR : [float]
        The bR (for each bias, current etc.)
        
	Returns
	-------
	sigmak_des : np.array([float])
        The uncertatinty on the desorbtion rate (for each bias, current etc.)
    sigmak_diff : np.array([float])
        The uncertatinty on the diffusion rate (for each bias, current etc.)  
    """
    sigmakDiffFit = np.array(sigmakDiffFit)
    sigmakDesFit = np.array(sigmakDesFit)
    kDiff = np.array(kDiff)
    kDes = np.array(kDes)
    sigmaBR = np.array(sigmaBR)
    bR = np.array(bR)
    
    # sigmak_des = sqrt( (sigmak_diff*B)^2 + (sigmaB*k_diff)^2 )
    sigmak_des = np.sqrt( (sigmakDiffFit*bR)**2 + (sigmaBR*kDiff)**2 )
    # sigmak_diff = sqrt( (sigmak_des/B)^2 + (sigmaB*k_des/B^2)^2 )
    sigmak_diff = np.sqrt( (sigmakDesFit/bR)**2 + (sigmaBR*kDes/(bR**2))**2 )
    return sigmak_des, sigmak_diff

def thermal_model(V, *popt):
    """Fit to the bRatio for a given bias below the non-local region with the Arrhenius model.
    Fixed parameters from other papers are defined inside the function.
    Energy units are eV
    
    Arguments
	---------
	V : np.array([float])
		The bias of the experiments
    E0 : float
        The base thermal energy of the physisorbed state
    alpha : float
        Dimensionless parameter describing the efficiency of converting excess electron energy into thermal energy
    
	Returns
	-------
	B : np.array([float])
       The expected bRatio in the model for a given bias
    """
    # A = 1.176e3 # Ratio of Arrhenius prefactors (des/diff) from DL
    A = 1.2e3 # rounding the above PAPER
    # A = 10e10 # from TK
    deltaE = 0.22 # Energy gained by system in a DIET cycle from DL PAPER
    # deltaE = 0.342 # from TK
    V_Th = 1.4 # the manipulation turn on threshold
    Es = 0.35 # the barrier to switching from physisorbed state
    E0, alpha = popt
    # B = A * np.exp(-1*deltaE/(E0+alpha*(V-V0)))
    B = A * np.exp(-1*deltaE/(E0+alpha*(V-V_Th+Es)))
    return(B)

def des_percent(bR):
    """Calculate the % of mols desorbed from a given branching ratio
    
    Arguments
	---------
	bR : [float] or np.array([float])
		The branching ratio value(s) to be converted
        
	Returns
	-------
	100*bR/(bR+1) : np.array([float])
       The % of mols desorbed
    """
    bR = np.array(bR)
    return 100*bR/(bR+1)

def percent_to_bR(desP): # Need for secondary y axis to be shared
    """Calculate the branching ratio from a given % of mols desorbed 
    
    Arguments
	---------
	desP : [float] or np.array([float])
		The desorbed % value(s) to be converted
        
	Returns
	-------
	desP/(100-desP) : np.array([float])
       The branching ratio(s)
    """
    return desP/(100-desP)

def binomial_confidence(n, k, q=0.6875):
    """
    Calculates binomial confidence intervals using the beta distribution.
    See e.g. Cameron 2011, PASA, 28, 128
    @param n: number of tries
    @type n: number/array
    @param k: number of successes
    @type k: number/array
    @param q: confidence interval (i.e. 68% would be a 1sigma confidence interval)
    @type q: float
    @return: confidence limit at q,p hat,confidence limit at 1-q
    @rtype: array
    """
    if type(n) is int and type(k) is int:
        n = [n]
        k = [k]
    if len(n) != len(k):
        print("n and k must have same shape.")
        raise ValueError
    full_n = np.array(n, dtype=float)
    full_k = np.array(k, dtype=float)
    conf_low = np.zeros_like(full_n)
    conf_high = np.zeros_like(full_n)
    p_hat = np.zeros_like(full_n)
    j = 0
    for n, k in zip(full_n, full_k):
        if k == 0:
            p_hat[j] = 0
            conf_low[j] = 0
            conf_high[j] = beta.ppf(q, k+1, n-k+1)
        elif n == k:
            p_hat[j] = 1
            conf_low[j] = beta.ppf((1.-q), k+1, n-k+1)
            conf_high[j] = 1
        else:
            conf_low[j] = beta.ppf((1-q)/2., k+1, n-k+1)
            conf_high[j] = beta.ppf(1-(1-q)/2., k+1, n-k+1)
            p_hat[j] = 1.*k/n
        j += 1
    return(np.array([conf_low, p_hat, conf_high]))

def ceiling_division(n, d):
    """Perform ceiling divison - divide then round up - between two values. 
    
    Arguments
	---------
	n : int
		The numerator
    d : int
        The denominator
        
	Returns
	-------
	(n + d - 1) // d : int
       The result of the ceiling division
    """
    return (n + d - 1) // d

def round_to_uncertainty(value, uncert):
    """Round a value to the appropriate number of sig figs based on the associated uncertainty 
    
    Arguments
	---------
	value : float
		The value to be rounded
    uncert : float
        The uncertainty on the value
        
	Returns
	-------
	(val, unc) : (float, float)
       The rounded value and and uncertainty
    """
    unc = round(uncert, -int(floor(log10(abs(uncert)))))
    val = round(value, -int(floor(log10(abs(uncert)))))
    return (val, unc)

def adj_r2(actual, predicted, nparams):
    """Find the adjusted R2 value for given data based on the measured & predicted values and the number of predictor parameters
    adjR2 = 1- (1-R2)*(n-1)/(n-p-1)
    
    Arguments
	---------
	actual : int or float
		The measured value
    predicted : int or float
        The value predicted by the model
    nparams : int
        The number of predictor variables in the model
        
	Returns
	-------
	1-(1-r2_score(actual, predicted))*(len(actual)-1)/(len(actual)-nparams-1) : float
       The adjusted R2 value
    """
    if len(actual)-nparams-1 == 0: # deal with the 2nd data point n=2 case
        return np.nan
    return 1-(1-r2_score(actual, predicted))*(len(actual)-1)/(len(actual)-nparams-1)

def I_captured(integrationVoltages, fitParams): # omega
    """
    Calculate the total charge captured by a state based on tunneling transmission and state capture probability.

    This function integrates the product of the tunneling transmission coefficient (`T`) and the capture probability (`G`)
    over the specified integration voltage points. The result represents the total charge captured by the state 
    defined by a given set of state fit parameters at a given injection energy.
    
    Most of this theory based on Nat. Comms. 2016 - Initiating and imaging the coherent surfacedynamics...
    and Nanoscale Adv. 2022 - A self-consistent model to link surface...
    
    This integral is called \Omega in Sloan's notation 

    Arguments
    ---------
    integrationVoltages : np.array([float])
        Voltage points over which the integration is performed. 
        The final voltage point represents the injection bias, i.e., the maximum voltage to integrate up to
        
    fitParams : [float]
        The Gaussian parameters for each state.

    Returns
    -------
    totalCaptured : float
        The total charge captured by the state at the given injection energy.
    """
    
    z0 = 6e-10 # sensible tunnelling gap (m)
    E_v = 4.6 # Vacuum energy from Nat. Comms. paper (eV)
    e = 1.6e-19 # electron charge
    m_e = 9.11e-31
    hbar = 1e-34
    
    V_i = integrationVoltages[-1] # injection bias - the maximum point we want to integrate up to
    mu, sigma, A = fitParams
    
    # I captured = integral[T*G]
    T = np.exp( -2 * z0 * np.sqrt(2*e*(m_e/hbar**2)*(E_v + V_i/2 - integrationVoltages)))
    G = A * np.exp( -1*(integrationVoltages-mu)**2 / (2*sigma**2))
    TG = T*G # the charge captured at each integration point for a given injection bias
    return TG.sum()

def frac_I_captured(integrationVpoints, stateParams): # s_i(V)
    """
    Calculate the fraction of the tunnelling current captured by each state at an injection bias given by each integration voltage point.
    Passes the required parameters to I_captured, then does the fraction arithmetic.
    Frac captured called s_i(V) in Sloan's notation

    Arguments
    ---------
    integrationVpoints : np.array([float])
        The bias points used for the integration. Defined in the primary function

    stateParams : [float]
        The Gaussian parameters for each state. Every three consecutive elements represent the parameters for one state
    
    Returns
    -------
    fracCaptured : np.array([float])
        A (2D) array where each row represents a state and each column represents the fractional 
        capture values at the corresponding integration voltage points
    """
    
    eachStateCaptured = [] # sVj
    for i in range(0, len(stateParams), 3): #loop over each state
        stateCaptured = [] # holds total charge captured by the state at the given tunnel bias
        for injV in integrationVpoints: # calulate for injecting at each integration point
            V_mask = (integrationVpoints <= injV)
            integrationVbelowInj = integrationVpoints[V_mask]
            stateCaptured.append(I_captured(integrationVbelowInj, stateParams[i:i+3]))
        eachStateCaptured.append(stateCaptured)

    eachStateCaptured = np.array(eachStateCaptured)
    totalCaptured = np.sum(eachStateCaptured, axis=0) # summing down the columns to give total capture at each integration point
    fracCaptured = np.array([stateCap / totalCaptured for stateCap in eachStateCaptured])
    return fracCaptured

def beta_param_log(voltage, *betas, fixed_fracCapturedArray):
    """
    Calculate the base 10 logarithm of the weighted sum (by beta) of fractional captures for each bias.
    --> log_10( \sum beta_i * s_i(V) )
    This sum at a bias is used to fit to the reaction rates at a given bias (c.f. voltage param not explicitly used in function)
    The function is intended to be used with `fixed_fracCapturedArray` passed as a fixed argument
    using `functools.partial` or a similar method.
    Fitting in log space can help make error bars similar magnitudes so don't need weighted fit.
    See below for fit in linear space.

    Arguments
    ---------
    voltage : [float]
        The voltage values at which the beta parameter is being calculated. (Used when fitting).
        
    betas : [float]
        Variable length argument list, where each `beta` is a weighting factor applied to the corresponding
        state in `fixed_fracCapturedArray`.
        
    fixed_fracCapturedArray : np.array([float])
        A (2D) array where each row represents a state and each column represents the fractional 
        capture values at the corresponding integration voltage points

    Returns
    -------
    np.log10(betaSum) : np.array([float])
        The base 10 logarithm of the weighted sum of fractional captures for each bias.
    """
    betaS = [] # hold all the beta_i * s_i(V) values
    for beta_i, state in zip(betas, fixed_fracCapturedArray):
        betaS.append(beta_i*state)
    betaSum = np.sum(betaS, axis=0) # summing down the columns at each bias
    return np.log10(betaSum)

def beta_param_lin(voltage, *betas, fixed_fracCapturedArray):
    """
    Calculate the weighted sum (by beta) of fractional captures for each bias.
    --> \sum beta_i * s_i(V)
    This sum at a bias is used to fit to the reaction rates at a given bias (c.f. voltage param not explicitly used in function)
    The function is intended to be used with `fixed_fracCapturedArray` passed as a fixed argument
    using `functools.partial` or a similar method.
    See above for fit in log space.

    Arguments
    ---------
    voltage : [float]
        The voltage values at which the beta parameter is being calculated. (Used when fitting).
        
    betas : [float]
        Variable length argument list, where each `beta` is a weighting factor applied to the corresponding
        state in `fixed_fracCapturedArray`.
        
    fixed_fracCapturedArray : np.array([float])
        A (2D) array where each row represents a state and each column represents the fractional 
        capture values at the corresponding integration voltage points

    Returns
    -------
    betaSum : np.array([float])
        The weighted sum of fractional captures for each bias.
    """
    betaS = [] # hold all the beta_i * s_i(V) values
    for beta_i, state in zip(betas, fixed_fracCapturedArray):
        betaS.append(beta_i*state)
    betaSum = np.sum(betaS, axis=0) # summing down the columns at each bias
    return betaSum

def match_V_to_frac_captured(integrationVpoints, targetVoltages, fracCapturedArray):
    """
    Match the closest integration voltage points to the target voltages and retrieve the corresponding fractional capture values for each state.
    
    Arguments
    ----------
    integrationVpoints : np.array([float])
        The bias points used for the integration. Defined in the primary function
        
    targetVoltages : np.array([float])
        Target voltage points for which the closest integration points need to be found
        
    fracCapturedArray : np.array([float])
        A (2D) array where each row represents a state and each column represents the fractional 
        capture values at the corresponding integration voltage points

    Returns
    -------
    eachStateFracCapAtTargetV : np.array([float])
        A (2D) array where each row corresponds to a state and each column contains the fractional 
        capture values at the closest integration voltage points to the `targetVoltages`
    """
    eachStateFracCapAtTargetV = []
    
    for state in fracCapturedArray:
        stateFracCapturedTargetV = []
        for voltage in targetVoltages:
            closeIndex = (np.abs(integrationVpoints - voltage)).argmin()
            stateFracCapturedTargetV.append(state[closeIndex])
        eachStateFracCapAtTargetV.append(stateFracCapturedTargetV)
        
    return np.asarray(eachStateFracCapAtTargetV)

def state_colour(stateE):
    """
    Determine the color associated with a given state energy value.
    Maps specific energy ranges corresponding to the LUMO and the U2 nonL state to colours 
    from the `gnuplot2` colormap otherwise a default color of grey is returned.

    Arguments
    ----------
    stateE : float
        The state energy value to be evaluated.

    Returns
    -------
    colour : str or np.array([float])
        A color from the `gnuplot2` colormap as an array if `stateE` falls within the 
        predefined ranges. If `stateE` does not match any range, the color string 'grey' is returned.
        
        - If 1.6 < stateE < 1.8, returns the color corresponding to `colours[2]`.
        - If 2.1 < stateE < 2.3, returns the color corresponding to `colours[3]`.
        - Otherwise, returns 'grey'.
    """
    colours = plt.cm.gnuplot2(np.linspace(0, 0.9, 6))
    if 1.6 < stateE < 1.8:
        colour = colours[2]
    elif 2.2 < stateE < 2.3:
        colour = colours[3]
    else:
        colour = 'grey'
    return colour 

def extract_data(fileDes, fileDiff, variable):
    """Turn csv of manipulation times at each bias or current etc. into DataFrame in usable format
    
    Arguments
	---------
	fileDes : str
		The csv containing the desorbtion (switching) data
    fileDiff : str
        The csv containing the diffusion data
    variable : str
        The heading that should be used for the variable of interest
	
    Returns
	-------
    columns : [str]
        The column headings - usually the bias, current etc. of the experiments
	df_manipulated : pd.DataFrame
       The processed DataFrame
    """
    des_data = pd.read_csv(fileDes)
    diff_data = pd.read_csv(fileDiff)
    
    columns = list(des_data) # grabs the header
    columns_match = list(diff_data)
    if columns != columns_match:
        print('\n\n### Column headers do not match - exiting ###\n\n')
        sys.exit(1)
    
    times_dict = {} # hold time of every event for each column e.g. {bias V:[evt times]}
    for column in columns:
        timesList = des_data[column].tolist() # get the des data
        timesList.extend(diff_data[column].tolist()) # add the diff data
        # Remove NaNs from list and order. Calibrate 0 time to first manip evt if needed.
        timesList = [time for time in timesList if np.isfinite(time)]
        timesList.sort()
        minTime = min(timesList)
        if minTime > 0: # times not calibrated to first manip evt.
            print('Calibrating time data')
            timesList = [time - minTime for time in timesList]
            des_data[column] = [time - minTime for time in des_data[column]]
            diff_data[column] = [time - minTime for time in diff_data[column]]
        else:
            print('Data already time calibrated')
        
        # Finally adding to times dict.
        times_dict[column] = timesList
    
    # Make a dict to hold data that will then be turned into a df 
    manipulated_dict = {key:[] for key in [f'{variable}', 'Time', 'nDes', 'nDiff', 'nManip']}
    # Loop over each time for each bias and count n evts in each case
    for column in times_dict:
        for time in times_dict[column]:
            nDes = np.count_nonzero(des_data[column] <= time)
            nDiff = np.count_nonzero(diff_data[column] <= time)
            nManip = nDes + nDiff
            manipulated_dict[f'{variable}'].append(column)
            manipulated_dict['Time'].append(time)
            manipulated_dict['nDes'].append(nDes)
            manipulated_dict['nDiff'].append(nDiff)
            manipulated_dict['nManip'].append(nManip)
    
    df_manipulated = pd.DataFrame.from_dict(manipulated_dict, orient='columns')
    
    return(columns, df_manipulated)

def branchRatioBias(voltages, df_manip):
    
    print(voltages)
    
    # dirty n stayed didct to add to N0
    nStayedList = [267, 44, 94, 24, 85, 50, 42, 75, 69]
    stayedDict = dict(zip(voltages, nStayedList))
    
    # Hold total nDes, nDiff and the calculated branching ratios
    nDesTot = []
    nDiffTot = []
    
    bRatio_linear_outer = []
    bRatio_doubleFit_outer = []
    
    k_des_outer = []
    k_diff_outer = []
    K_Tot_outer = []
    
    # Hold fit uncerts
    uncert_fit_linear = []
    uncert_kdes_rates_outer = []
    uncert_kdiff_rates_outer = []
    uncert_KTot_rates_outer = []
    uncert_KTot_dbF_outer = []
    uncert_kdes_dbF_outer = []
    uncert_kdiff_dbF_outer = []
    
    # Setup colourmap colour for each voltage
    colours = plt.cm.jet(np.linspace(0.0, 0.9, len(voltages)))
    
    # Setup the subplot grids for plots separated by bias
    # Choose always 3 cols, ceiling division to calculate n rows required
    plt.figure(7)
    fig_GoF_lin, axes_GoF_lin = plt.subplots(3, ceiling_division(len(voltages), 3), figsize=(15,10))
    axes_GoF_lin = axes_GoF_lin.ravel()
    plt.figure(8)
    fig_GoF_KTot, axes_GoF_KTot = plt.subplots(3, ceiling_division(len(voltages), 3), figsize=(15,10))
    axes_GoF_KTot = axes_GoF_KTot.ravel()
    
    plt.figure(123)
    fig_dbF_rates, axes_dbF_rates = plt.subplots(3, ceiling_division(len(voltages), 3), figsize=(10,10))
    axes_dbF_rates = axes_dbF_rates.ravel()
    
    # Find the rate and BR for each bias
    for voltage, colour, subPltIndex in zip(voltages, colours, range(len(voltages)+1)):
        print(f'Looking at {voltage}')
        
        # Set up a mask for the bias
        v_mask = (df_manip['Voltage']==voltage)
        df_manip_masked = df_manip[v_mask]
        
        # Extract final counts of des and diff
        ndestot = df_manip_masked['nDes'].iloc[-1]
        ndifftot = df_manip_masked['nDiff'].iloc[-1]
        N_0 = ndestot + ndifftot + stayedDict[voltage] # Toggle adding stayed dict on/off
        nDesTot.append(ndestot)
        nDiffTot.append(ndifftot)
        
        # Perform linear fit for BR and add to outer lists
        bR_lin, uncert_fit_lin = bR_linear(df_manip_masked)
        bRatio_linear_outer.append(bR_lin)
        uncert_fit_linear.append(uncert_fit_lin)
        # Plot the data with the fit
        xpoints = np.linspace(0, max(df_manip_masked['nDiff']))
        plt.figure(1)
        plt.plot(df_manip_masked['nDiff'], df_manip_masked['nDes'], 'x', color=colour)
        plt.plot(xpoints, line(xpoints, bR_lin), '--', color=colour)
        
        # Find the rates for each channel needed for part a) of the figure
        k_des, k_diff, K_Tot, uncert_tup_rates = path_rates(df_manip_masked, N_0)
        k_des_outer.append(k_des)
        k_diff_outer.append(k_diff)
        K_Tot_outer.append(K_Tot)
        uncert_kdes_rates_outer.append(uncert_tup_rates[0])
        uncert_kdiff_rates_outer.append(uncert_tup_rates[1])
        uncert_KTot_rates_outer.append(uncert_tup_rates[2])
        
        # Get the BR from double fit method
        bR_dbF, uncerts_tup = bR_doubleFit(df_manip_masked, N_0)
        bRatio_doubleFit_outer.append(bR_dbF)
        uncert_KTot_dbF_outer.append(uncerts_tup[0])
        uncert_kdes_dbF_outer.append(uncerts_tup[1])
        uncert_kdiff_dbF_outer.append(uncerts_tup[2])
        
        # Now with the loop over N manip - could combine into one loop and take last value for output.
        # Would save code space but not muc compute time - only 1 duplication so maybe less clear to reader?
        # Hold bR (or kx) & GoF as function of n mols manipulated for a given bias
        nManip = []
        bR_lin_n = []
        KTot_n = []
        bR_dbF_n = []
        uncert_fit_lin_n = []
        uncert_KTot_n = []
        
        adj_r2_GoF_lin_n = []
        adj_r2_GoF_KTot_n = []
        adj_r2_GoF_dbF_n = []
        
        for nMan in df_manip_masked['nManip']:
            n_mask = (df_manip_masked['nManip'] <= nMan)
            nManip.append(nMan) # not really needed to be this explicit but clear for now 
            
            # First for linear fit
            bR_n, uncert_n = bR_linear(df_manip_masked[n_mask])
            bR_lin_n.append(bR_n)
            uncert_fit_lin_n.append(uncert_n)
            # Looking at R2 of fit so how far away nDes and nDiff are from linear fit.
            # Prediction for y axis value (nDes) is thus bR*nDiff  w/ ndof =1
            GoF = adj_r2(df_manip_masked[n_mask]['nDes'], bR_n*df_manip_masked[n_mask]['nDiff'], 1)
            adj_r2_GoF_lin_n.append(GoF)
            
            # For fit to total rate 
            total_rate_fit_N0 = partial(total_rate_fit, N0=N_0)
            popt, pcov = curve_fit(total_rate_fit_N0, df_manip_masked[n_mask]['Time'], df_manip_masked[n_mask]['nManip'], p0=[1.0])
            K_Tot = popt[0]
            uncert_KTot = pcov[0,0]**0.5
            KTot_n.append(K_Tot)
            uncert_KTot_n.append(uncert_KTot)
            # Prediction for y axis (nManip) is KTot*Time w/ ndof = 1
            GoF = adj_r2(df_manip_masked[n_mask]['nManip'], total_rate_fit(df_manip_masked[n_mask]['Time'], K_Tot, N_0), 1)
            adj_r2_GoF_KTot_n.append(GoF)
        # end loop over N manipulated
            
            
        # Plot GoF and bR from linear fit method for the bias on the relevant subplot. 
        # Don't plot first 10 points as fit v. poor in that range so zooms out y axis.
        plt.figure(fig_GoF_lin)
        axes_GoF_lin[subPltIndex].plot(nManip[10:], bR_lin_n[10:], 'ok', mfc='none', markersize=3)
        axes_GoF_lin[subPltIndex].set_xlabel('N mols manipulated')
        axes_GoF_lin[subPltIndex].set_ylabel(r'$k_d$/$k_s$')
        # Add simple division BR to plot 
        axes_GoF_lin[subPltIndex].plot(nManip[10:], df_manip_masked['nDes'][10:]/df_manip_masked['nDiff'][10:], 'rX', mfc='none', markersize=3)
        ax2 = axes_GoF_lin[subPltIndex].twinx()
        ax2.plot(nManip[10:], adj_r2_GoF_lin_n[10:], 'sb', mfc='none', markersize=3)
        ax2.set_ylabel(r'Adj. $R^2$', color='b')
        ax2.set_ylim(top=1)
        ax2.tick_params('y', colors='b')
        axes_GoF_lin[subPltIndex].set_title(voltage)
        
        if voltage == '2.1 V':
            plt.figure('2.1V_GOF')
            plt.plot(nManip[10:], bR_lin_n[10:], 'sk', mfc='none', markersize=3)
            plt.xlabel('N mols manipulated')
            plt.ylabel(r'$k_d$/$k_s$')
            # do the ticks for the first axis
            plt.tick_params(direction='in', length=4,            # A reasonable length
                bottom=True, left=True, top=True, right=True)
            ax = plt.gca()
            ax2 = ax.twinx()
            ax2.plot(nManip[10:], adj_r2_GoF_lin_n[10:], 'ob', mfc='none', markersize=3)
            ax2.set_ylabel(r'Adj. $R^2$', color='b')
            ax2.set_ylim(top=1)
            # and the 2ndary axis ticks
            ax2.tick_params('y', colors='b', direction='in', length=4)
            
            plt.savefig('figures/GoF_bRatioLinearFit_2-1V.png')
            plt.savefig('figures/GoF_bRatioLinearFit_2-1V.pdf')
            plt.close()
            
        # Plot GoF and KTot for the bias on the subplots.
        # Same cut on first points?
        plt.figure(fig_GoF_KTot)
        axes_GoF_KTot[subPltIndex].plot(nManip[10:], KTot_n[10:], 'ok', mfc='none', markersize=3)
        axes_GoF_KTot[subPltIndex].set_xlabel('N mols manipulated')
        axes_GoF_KTot[subPltIndex].set_ylabel(r'$K_{Tot}$')
        ax2 = axes_GoF_KTot[subPltIndex].twinx()
        ax2.plot(nManip[10:], adj_r2_GoF_KTot_n[10:], 'sb', mfc='none', markersize=3)
        ax2.set_ylabel(r'Adj. $R^2$', color='b')
        ax2.set_ylim(0.8, 1)
        ax2.tick_params('y', colors='b')
        axes_GoF_KTot[subPltIndex].set_title(voltage)
        
        # plot separate and combined rates from dbF method for sup figure
        # ** N.B. need to toggle N stayed dict off for this plot -> maths works with nManip not N0 but easier to keep rest of code same
        plt.figure(fig_dbF_rates)
        # first the n over time
        axes_dbF_rates[subPltIndex].plot(df_manip_masked['Time'], df_manip_masked['nDes'], 'oc', mfc='none', markersize=2)
        axes_dbF_rates[subPltIndex].plot(df_manip_masked['Time'], df_manip_masked['nDiff'], 'oy', mfc='none', markersize=2)
        axes_dbF_rates[subPltIndex].plot(df_manip_masked['Time'], df_manip_masked['nManip'], 'ok', mfc='none', markersize=2)
        # then the fits
        axes_dbF_rates[subPltIndex].plot(df_manip_masked['Time'], kx_fit(df_manip_masked['Time'], k_des, N_0, K_Tot), '-c', markersize=2)
        axes_dbF_rates[subPltIndex].plot(df_manip_masked['Time'], kx_fit(df_manip_masked['Time'], k_diff, N_0, K_Tot), '-y', markersize=2)
        axes_dbF_rates[subPltIndex].plot(df_manip_masked['Time'], total_rate_fit(df_manip_masked['Time'], K_Tot, N_0), '-k', markersize=2)
        axes_dbF_rates[subPltIndex].set_xlabel('Time (s)')
        axes_dbF_rates[subPltIndex].set_ylabel(r'N mols manipulated')
        axes_dbF_rates[subPltIndex].set_title(voltage)
        # now deal with the ticks
        axes_dbF_rates[subPltIndex].tick_params(direction='in', length=4,            # A reasonable length
                bottom=True, left=True, top=True, right=True)
        
        # Plot pdf for beta dist
        if float(voltage.split(' ')[0]) < 2:
            plt.figure('fig_beta_pdf')
            x = np.linspace(beta.ppf(0.001, ndestot+1, N_0-ndestot+1), 
                            beta.ppf(0.999, ndestot+1, N_0-ndestot+1), 100)
            plt.plot(x, beta.pdf(x, ndestot+1, N_0-ndestot+1), color=colour, label=voltage)
            plt.fill_between(x, beta.pdf(x, ndestot+1, N_0-ndestot+1), color=colour, alpha=0.3)
            # # area under curve to norm to unity?
            # aUnder = auc(percent_to_bR(100*x), beta.pdf(x, ndestot+1, N_0-ndestot+1))
            # plt.plot(x, beta.pdf(x, ndestot+1, N_0-ndestot+1)/aUnder, color=colour, label=voltage)
            # plt.fill_between(x, beta.pdf(x, ndestot+1, N_0-ndestot+1)/aUnder, color=colour, alpha=0.3)
            
            # # Converted to bR from fraction desorbed
            # plt.plot(percent_to_bR(100*x), beta.pdf(x, ndestot+1, N_0-ndestot+1), color=colour, label=voltage)
            # plt.fill_between(percent_to_bR(100*x), beta.pdf(x, ndestot+1, N_0-ndestot+1), color=colour, alpha=0.3)
            # area under curve to norm to unity?
            # aUnder = auc(percent_to_bR(100*x), beta.pdf(x, ndestot+1, N_0-ndestot+1))
            # plt.plot(percent_to_bR(100*x), beta.pdf(x, ndestot+1, N_0-ndestot+1)/aUnder, color=colour, label=voltage)
            # plt.fill_between(percent_to_bR(100*x), beta.pdf(x, ndestot+1, N_0-ndestot+1)/aUnder, color=colour, alpha=0.3)
    # end loop over voltages
        
    # Find the final point BR
    bRatio_finalPoint = [ndes/ndiff for ndes, ndiff in zip(nDesTot, nDiffTot)]
    # Calculate total manipulated
    nManipTot = [ndes + ndiff for ndes, ndiff in zip(nDesTot, nDiffTot)]
    
    # Plot the raw counts
    plt.figure(1)
    plt.xlabel('Number of Diffused Molecules')
    plt.ylabel('Number of Desorbed Molecules')
    plt.savefig('figures/nDesDiffRawLinearFitBias.png')
    plt.close()
    
    # Counting stats uncert with np.array arithmatic on the lists
    # sigmaB = B * sqrt((sqrtD/D)^2 + (sqrtS/S)^2)
    sigmaB_linear = bRatio_linear_outer * np.sqrt( (np.sqrt(nDesTot)/nDesTot)**2 + (np.sqrt(nDiffTot)/nDiffTot)**2 )
    binomial_beta_props = binomial_confidence(nManipTot, nDesTot)
    # Get lower & upper bounds from beta dist and convert from frac -> % -> bR
    beta_limts = [percent_to_bR(100*binomial_beta_props[0]), percent_to_bR(100*binomial_beta_props[2])]
    sigmaB_beta = [[bR - lowL for bR, lowL in zip(bRatio_linear_outer, beta_limts[0])], [uppL - bR for bR, uppL in zip(bRatio_linear_outer, beta_limts[1])]]
    sigmaB_beta_finalP = [[bR - lowL for bR, lowL in zip(bRatio_finalPoint, beta_limts[0])], [uppL - bR for bR, uppL in zip(bRatio_finalPoint, beta_limts[1])]]
    # Binomial based uncerts sigmaB = sqrt(B/N + B^3/N)
    sigmaB_binomial = np.sqrt(np.array(bRatio_linear_outer)/nManipTot + np.array(bRatio_linear_outer)**3/nManipTot)
    
    # bRatio from linear fit
    # first fit to the thermal model 
    voltages = np.array([float(v.split(' ')[0]) for v in voltages]) # strip out the bias value from the string and make array
    # select the below nonL biases for fitting
    nLocThresh = 2.0 # the upper threshold of the below non-local region
    threshMask = (voltages < nLocThresh)
    p0 = [0.02, 0.01] #[Eo, alpha]
    popt, pcov = curve_fit(thermal_model, voltages[threshMask], np.array(bRatio_linear_outer)[threshMask], p0, maxfev=10000)
    print(f'popt E0: {round_to_uncertainty(popt[0], pcov[0,0]**0.5)[0]} \u00B1 {round_to_uncertainty(pcov[0,0]**0.5, pcov[0,0]**0.5)[1]}\n popt alpha: {round_to_uncertainty(popt[1], pcov[1,1]**0.5)[0]} \u00B1 {round_to_uncertainty(pcov[1,1]**0.5, pcov[1,1]**0.5)[1]}\n')
    # then plot the bRatio
    plt.figure(2, dpi=600)
    ax = plt.gca()
    ax2 = ax.secondary_yaxis("right", functions=(des_percent, percent_to_bR))
    # plt.errorbar(voltages, bRatio_linear_outer, yerr = sigmaB_linear, fmt='ks', markersize=4, capsize=4, capthick=1)
    # plt.errorbar(voltages, bRatio_linear_outer, yerr = sigmaB_beta, fmt='bs', markersize=4, capsize=4, capthick=1)
    plt.errorbar(voltages, bRatio_linear_outer, yerr = sigmaB_binomial, fmt='ks', markersize=4, capsize=4, capthick=1)
    # add the fit to the plot
    xpoints = np.linspace(min(voltages[threshMask]), max(voltages[threshMask]), 500)
    plt.plot(xpoints, thermal_model(xpoints, popt[0], popt[1]), 'k--')
    plt.xlim(1.3, 2.3)
    plt.ylim(1.5, 8)
    plt.xlabel('Sample bias (V)')
    plt.ylabel(r'$k_d$/$k_s$')
    ax2.set_ylabel('Desorbed %')
    # now deal with the ticks
    plt.tick_params(direction='in', length=4,            # A reasonable length
                bottom=True, left=True, top=True, right=False) # want these (inwards on 3 axes) ticks on 3 axes. Want other (outwards) on 2ndary y axis
    ax = plt.gca()
    xloc = ticker.MultipleLocator(base=0.1) # this locator puts ticks at regular intervals
    ax.xaxis.set_major_locator(xloc)
    yloc = ticker.MultipleLocator(base=1.0) # this locator puts ticks at regular intervals
    ax.yaxis.set_major_locator(yloc)
    n = 2  # Keeps every 2nd tick label - use 0 or 1 at end next line for where want start
    [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if i % n != 0]
    [l.set_visible(False) for (i,l) in enumerate(ax.yaxis.get_ticklabels()) if i % n != 1]
    plt.savefig('figures/bRatioLinearFitBias.png')
    plt.savefig('figures/bRatioLinearFitBias.pdf')
    plt.close()
    
    # the rates for each reaction path (part (a) of the figure)
    sigmak_des, sigmak_diff = path_rates_errs(uncert_kdiff_rates_outer, uncert_kdes_rates_outer, k_diff_outer, k_des_outer, sigmaB_binomial, bRatio_linear_outer)
    plt.figure(11, dpi=600)
    # plt.yscale('log')
    plt.errorbar(voltages, k_des_outer, yerr = sigmak_des, fmt='ko', markersize=4, capsize=4, capthick=1, label='Desorbed', mfc='white')
    plt.errorbar(voltages, k_diff_outer, yerr = sigmak_diff, fmt='ko', markersize=4, capsize=4, capthick=1, label='Switched')
    plt.xlim(1.3, 2.3)
    # plt.ylim(0, 3.5)
    plt.ylim(0.01, 5)
    plt.xlabel('Sample bias (V)')
    plt.ylabel(r'Rate of manipulation ($s^{-1}$)')
    plt.legend(loc='upper left')
    # now deal with the ticks
    plt.tick_params(direction='in', which='both', length=4,            # A reasonable length
                bottom=True, left=True, top=True, right=True)
    ax = plt.gca()
    xloc = ticker.MultipleLocator(base=0.1) # this locator puts ticks at regular intervals
    ax.xaxis.set_major_locator(xloc)
    # yloc = ticker.MultipleLocator(base=1) # this locator puts ticks at regular intervals
    # ax.yaxis.set_major_locator(yloc)
    n = 2  # Keeps every 2nd tick label - use 0 or 1 at end next line for where want start
    [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if i % n != 0]
    # [l.set_visible(False) for (i,l) in enumerate(ax.yaxis.get_ticklabels()) if i % n != 1]
    plt.savefig('figures/pathRatesBias.png')
    plt.savefig('figures/pathRatesBias.pdf')
    plt.yscale('log')
    plt.savefig('figures/pathRatesBiasLog.pdf')
    plt.savefig('figures/pathRatesBiasLog.png')
    plt.close()
    
    # bRatio from final points
    plt.figure(3)
    ax = plt.gca()
    ax2 = ax.secondary_yaxis("right", functions=(des_percent, percent_to_bR))
    plt.errorbar(voltages, bRatio_finalPoint, yerr = sigmaB_beta_finalP, fmt='ks', markersize=4, capsize=4, capthick=1)
    plt.xlabel('Sample bias (V)')
    plt.ylabel(r'$k_d$/$k_s$')
    ax2.set_ylabel('Desorbed %')
    plt.savefig('figures/bRatioFinalPointBias.png')
    plt.close()
    
    # Residual linearFit - finalPoint
    plt.figure(4)
    # ax = plt.gca()
    # ax2 = ax.secondary_yaxis("right", functions=(des_percent, percent_to_bR))
    plt.plot(voltages, [bR_lin - bR_fin for bR_lin, bR_fin in zip(bRatio_linear_outer, bRatio_finalPoint)], 'ko')
    plt.xlabel('Sample bias (V)')
    plt.ylabel('Residual (Linear fit - final point)')
    # ax2.set_ylabel('Desorbed %')
    plt.savefig('figures/residualLinear_FinalPointBias.png')
    plt.close()
    
    # Uncorrelated dBF uncert
    # sigmaB = DeltakDes/kDes + DeltakDiff/kDiff
    sigmaB_dBF_uncorrelated = [Ddes + Ddiff for Ddes, Ddiff in zip(uncert_kdes_dbF_outer, uncert_kdiff_dbF_outer)]
    # bRatio from double fit
    plt.figure(5)
    ax = plt.gca()
    ax2 = ax.secondary_yaxis("right", functions=(des_percent, percent_to_bR))
    # plt.plot(voltages, bRatio_doubleFit_outer, 'ks')
    plt.errorbar(voltages, bRatio_doubleFit_outer, yerr=sigmaB_dBF_uncorrelated, fmt='ks', markersize=4, capsize=4, capthick=1)
    plt.xlabel('Sample bias (V)')
    plt.ylabel(r'$k_d$/$k_s$')
    ax2.set_ylabel('Desorbed %')
    plt.ylim(1.5, 8.5)
    plt.savefig('figures/bRatioDoubleFitBias.png')
    plt.close()
    
    # Residual linearFit - finalPoint
    plt.figure(6)
    # ax = plt.gca()
    # ax2 = ax.secondary_yaxis("right", functions=(des_percent, percent_to_bR))
    plt.plot(voltages, [bR_lin - bR_dbF for bR_lin, bR_dbF in zip(bRatio_linear_outer, bRatio_doubleFit_outer)], 'ko')
    plt.xlabel('Sample bias (V)')
    plt.ylabel('Residual (Linear fit - double fit)')
    # ax2.set_ylabel(r'\Delta Desorbed %')
    plt.savefig('figures/residualLinear_doubleFitBias.png')
    plt.close()
    
    plt.figure(fig_GoF_lin)
    fig_GoF_lin.subplots_adjust(wspace=0.5, hspace=0.4) # adjust width spacing so 2nd y axis not overlapping
    plt.tight_layout()
    plt.savefig('figures/GoF_bRatioLinearFitBias.png')
    plt.close()
    
    plt.figure(fig_GoF_KTot)
    fig_GoF_lin.subplots_adjust(wspace=0.5, hspace=0.4) # adjust width spacing so 2nd y axis not overlapping
    plt.tight_layout()
    plt.savefig('figures/GoF_KTotBias.png')
    plt.close()
    
    plt.figure(fig_dbF_rates)
    plt.tight_layout()
    plt.savefig('figures/dbF_ratesBias.png')
    plt.savefig('figures/dbF_ratesBias.pdf')
    plt.close()
    
    plt.figure('fig_beta_pdf')
    plt.xlabel(r'$k_{des}/k_{diff}$')
    plt.ylabel('pdf')
    plt.legend()
    plt.savefig('figures/beta_pdfsBias.png')
    plt.close()
    
    # Pull togther rates data needed for the STS fitting including calculating binomial error on K_Tot
    ratesTuple = ( voltages, K_Tot_outer, np.sqrt(np.array(sigmak_des)**2 + np.array(sigmak_diff)**2) )
    return ratesTuple

def branchRatioCurrent(currents, df_manip):
    
    print(currents)
    
    # dirty n stayed didct to add to N0
    nStayedList = [419, 193, 144, 127, 6, 190, 107, 227, 307]
    stayedDict = dict(zip(currents, nStayedList))
    
    # Reorder the currents by value
    currents.sort(key = lambda I: float(I.split(' ')[0]))
    print(f'sorted currents: {currents}')
    
    # Hold total nDes, nDiff and the calculated branching ratios
    nDesTot = []
    nDiffTot = []
    
    bRatio_linear_outer = []
    bRatio_doubleFit_outer = []
    
    k_des_outer = []
    k_diff_outer = []
    K_Tot_outer = []
    
    # Hold fit uncerts
    uncert_fit_linear = []
    uncert_kdes_rates_outer = []
    uncert_kdiff_rates_outer = []
    uncert_KTot_rates_outer = []
    uncert_KTot_dbF_outer = []
    uncert_kdes_dbF_outer = []
    uncert_kdiff_dbF_outer = []
    
    # Setup colourmap colour for each current
    colours = plt.cm.jet(np.linspace(0.0, 0.9, len(currents)))
    
    # Setup the subplot grids for plots separated by bias
    # Choose always 3 cols, ceiling division to calculate n rows required
    plt.figure(7)
    fig_GoF_lin, axes_GoF_lin = plt.subplots(3, ceiling_division(len(currents), 3), figsize=(15,10))
    axes_GoF_lin = axes_GoF_lin.ravel()
    plt.figure(8)
    fig_GoF_KTot, axes_GoF_KTot = plt.subplots(3, ceiling_division(len(currents), 3), figsize=(15,10))
    axes_GoF_KTot = axes_GoF_KTot.ravel()
    
    # choose which currenst to look at, e.g. include 200 pA or not
    selected_currents = ['5.00E-11 A', '1.50E-10 A', '2.00E-10 A', '3.00E-10 A', '4.50E-10 A', '6.00E-10 A', '7.50E-10 A', '9.00E-10 A']
    # selected_currents = ['2.50E-11 A', '5.00E-11 A', '1.50E-10 A', '2.00E-10 A', '3.00E-10 A', '4.50E-10 A', '6.00E-10 A', '7.50E-10 A', '9.00E-10 A']
    # selected_currents = ['5.00E-11 A', '1.50E-10 A', '3.00E-10 A', '4.50E-10 A', '6.00E-10 A', '7.50E-10 A', '9.00E-10 A']
    
    # Find the rate and BR for each current
    for current, colour, subPltIndex in zip(currents, colours, range(len(currents)+1)):
        print(f'Looking at {current}')
        
        # Set up a mask for the current
        I_mask = (df_manip['Current']==current) & (df_manip['Current'].isin(selected_currents))
        # selectI = df_manip['Current'].isin(selected_currents)
        df_manip_masked = df_manip[I_mask]
        
        if df_manip_masked.empty:
            print('empty - exit')
            continue
        
        # Extract final counts of des and diff
        ndestot = df_manip_masked['nDes'].iloc[-1]
        ndifftot = df_manip_masked['nDiff'].iloc[-1]
        N_0 = ndestot + ndifftot + stayedDict[current]
        nDesTot.append(ndestot)
        nDiffTot.append(ndifftot)
        
        # Perform linear fit for BR and add to outer lists
        bR_lin, uncert_fit_lin = bR_linear(df_manip_masked)
        bRatio_linear_outer.append(bR_lin)
        uncert_fit_linear.append(uncert_fit_lin)
        # Plot the data with the fit
        xpoints = np.linspace(0, max(df_manip_masked['nDiff']))
        plt.figure(1)
        plt.plot(df_manip_masked['nDiff'], df_manip_masked['nDes'], 'x', color=colour)
        plt.plot(xpoints, line(xpoints, bR_lin), '--', color=colour)
        
        # Find the rates for each channel needed for part a) of the figure
        k_des, k_diff, K_Tot, uncert_tup_rates = path_rates(df_manip_masked, N_0)
        k_des_outer.append(k_des)
        k_diff_outer.append(k_diff)
        K_Tot_outer.append(K_Tot)
        uncert_kdes_rates_outer.append(uncert_tup_rates[0])
        uncert_kdiff_rates_outer.append(uncert_tup_rates[1])
        uncert_KTot_rates_outer.append(uncert_tup_rates[2])
        # and the tottal rate
        total_rate_fit_N0 = partial(total_rate_fit, N0=N_0)
        
        # Get the BR from double fit method
        bR_dbF, uncerts_tup = bR_doubleFit(df_manip_masked, N_0)
        bRatio_doubleFit_outer.append(bR_dbF)
        uncert_KTot_dbF_outer.append(uncerts_tup[0])
        uncert_kdes_dbF_outer.append(uncerts_tup[1])
        uncert_kdiff_dbF_outer.append(uncerts_tup[2])
        
        # Now with the loop over N manip - could combine into one loop and take last value for output.
        # Would save code space but not muc compute time - only 1 duplication so maybe less clear to reader?
        # Hold bR (or kx) & GoF as function of n mols manipulated for a given bias
        nManip = []
        bR_lin_n = []
        KTot_n = []
        bR_dbF_n = []
        uncert_fit_lin_n = []
        uncert_KTot_n = []
        
        adj_r2_GoF_lin_n = []
        adj_r2_GoF_KTot_n = []
        adj_r2_GoF_dbF_n = []
        
        for nMan in df_manip_masked['nManip']:
            n_mask = (df_manip_masked['nManip'] <= nMan)
            nManip.append(nMan) # not really needed to be this explicit but clear for now 
            
            # First for linear fit
            bR_n, uncert_n = bR_linear(df_manip_masked[n_mask])
            bR_lin_n.append(bR_n)
            uncert_fit_lin_n.append(uncert_n)
            # Looking at R2 of fit so how far away nDes and nDiff are from linear fit.
            # Prediction for y axis value (nDes) is thus bR*nDiff  w/ ndof =1
            GoF = adj_r2(df_manip_masked[n_mask]['nDes'], bR_n*df_manip_masked[n_mask]['nDiff'], 1)
            adj_r2_GoF_lin_n.append(GoF)
            
            # For fit to total rate 
            total_rate_fit_N0 = partial(total_rate_fit, N0=N_0)
            popt, pcov = curve_fit(total_rate_fit_N0, df_manip_masked[n_mask]['Time'], df_manip_masked[n_mask]['nManip'], p0=[1.0])
            K_Tot = popt[0]
            uncert_KTot = pcov[0,0]**0.5
            KTot_n.append(K_Tot)
            uncert_KTot_n.append(uncert_KTot)
            # Prediction for y axis (nManip) is KTot*Time w/ ndof = 1
            GoF = adj_r2(df_manip_masked[n_mask]['nManip'], total_rate_fit(df_manip_masked[n_mask]['Time'], K_Tot, N_0), 1)
            adj_r2_GoF_KTot_n.append(GoF)
        # end loop over N manipulated
            
            
        # Plot GoF and bR from linear fit method for the bias on the relevant subplot. 
        # Don't plot first 10 points as fit v. poor in that range so zooms out y axis.
        plt.figure(fig_GoF_lin)
        axes_GoF_lin[subPltIndex].plot(nManip[10:], bR_lin_n[10:], 'ok', mfc='none', markersize=3)
        axes_GoF_lin[subPltIndex].set_xlabel('N mols manipulated')
        axes_GoF_lin[subPltIndex].set_ylabel(r'$k_d$/$k_s$')
        # Add simple division BR to plot 
        axes_GoF_lin[subPltIndex].plot(nManip[10:], df_manip_masked['nDes'][10:]/df_manip_masked['nDiff'][10:], 'rX', mfc='none', markersize=3)
        ax2 = axes_GoF_lin[subPltIndex].twinx()
        ax2.plot(nManip[10:], adj_r2_GoF_lin_n[10:], 'sb', mfc='none', markersize=3)
        ax2.set_ylabel(r'Adj. $R^2$', color='b')
        ax2.set_ylim(top=1)
        ax2.tick_params('y', colors='b')
        axes_GoF_lin[subPltIndex].set_title(current)
        
        # Plot GoF and KTot for the bias on the subplots.
        # Same cut on first points?
        plt.figure(fig_GoF_KTot)
        axes_GoF_KTot[subPltIndex].plot(nManip[10:], KTot_n[10:], 'ok', mfc='none', markersize=3)
        axes_GoF_KTot[subPltIndex].set_xlabel('N mols manipulated')
        axes_GoF_KTot[subPltIndex].set_ylabel(r'$K_{Tot}$')
        ax2 = axes_GoF_KTot[subPltIndex].twinx()
        ax2.plot(nManip[10:], adj_r2_GoF_KTot_n[10:], 'sb', mfc='none', markersize=3)
        ax2.set_ylabel(r'Adj. $R^2$', color='b')
        ax2.set_ylim(0.8, 1)
        ax2.tick_params('y', colors='b')
        axes_GoF_KTot[subPltIndex].set_title(current)
        
        # Plot pdf for beta dist
        if float(current.split(' ')[0]) < 2:
            plt.figure('fig_beta_pdf')
            x = np.linspace(beta.ppf(0.001, ndestot+1, N_0-ndestot+1), 
                            beta.ppf(0.999, ndestot+1, N_0-ndestot+1), 100)
            plt.plot(x, beta.pdf(x, ndestot+1, N_0-ndestot+1), color=colour, label=current)
            plt.fill_between(x, beta.pdf(x, ndestot+1, N_0-ndestot+1), color=colour, alpha=0.3)
            # # area under curve to norm to unity?
            # aUnder = auc(percent_to_bR(100*x), beta.pdf(x, ndestot+1, N_0-ndestot+1))
            # plt.plot(x, beta.pdf(x, ndestot+1, N_0-ndestot+1)/aUnder, color=colour, label=current)
            # plt.fill_between(x, beta.pdf(x, ndestot+1, N_0-ndestot+1)/aUnder, color=colour, alpha=0.3)
            
            # # Converted to bR from fraction desorbed
            # plt.plot(percent_to_bR(100*x), beta.pdf(x, ndestot+1, N_0-ndestot+1), color=colour, label=current)
            # plt.fill_between(percent_to_bR(100*x), beta.pdf(x, ndestot+1, N_0-ndestot+1), color=colour, alpha=0.3)
            # area under curve to norm to unity?
            # aUnder = auc(percent_to_bR(100*x), beta.pdf(x, ndestot+1, N_0-ndestot+1))
            # plt.plot(percent_to_bR(100*x), beta.pdf(x, ndestot+1, N_0-ndestot+1)/aUnder, color=colour, label=current)
            # plt.fill_between(percent_to_bR(100*x), beta.pdf(x, ndestot+1, N_0-ndestot+1)/aUnder, color=colour, alpha=0.3)
    # end loop over currents
    
    # Convert currents [str] into pA
    currents = [float(I.split(' ')[0])*1e12 for I in currents if I in selected_currents]
    print(f'New currents: {currents}')
        
    # Find the final point BR
    bRatio_finalPoint = [ndes/ndiff for ndes, ndiff in zip(nDesTot, nDiffTot)]
    # Calculate total manipulated
    nManipTot = [ndes + ndiff for ndes, ndiff in zip(nDesTot, nDiffTot)]
    
    # Plot the raw counts
    plt.figure(1)
    plt.xlabel('Number of Diffused Molecules')
    plt.ylabel('Number of Desorbed Molecules')
    plt.savefig('figures/nDesDiffRawLinearFitCurrent.png')
    plt.close()
    
    # Counting stats uncert with np.array arithmatic on the lists
    # sigmaB = B * sqrt((sqrtD/D)^2 + (sqrtS/S)^2)
    sigmaB_linear = bRatio_linear_outer * np.sqrt( (np.sqrt(nDesTot)/nDesTot)**2 + (np.sqrt(nDiffTot)/nDiffTot)**2 )
    binomial_beta_props = binomial_confidence(nManipTot, nDesTot)
    # Get lower & upper bounds from beta dist and convert from frac -> % -> bR
    beta_limts = [percent_to_bR(100*binomial_beta_props[0]), percent_to_bR(100*binomial_beta_props[2])]
    sigmaB_beta = [[bR - lowL for bR, lowL in zip(bRatio_linear_outer, beta_limts[0])], [uppL - bR for bR, uppL in zip(bRatio_linear_outer, beta_limts[1])]]
    sigmaB_beta_finalP = [[bR - lowL for bR, lowL in zip(bRatio_finalPoint, beta_limts[0])], [uppL - bR for bR, uppL in zip(bRatio_finalPoint, beta_limts[1])]]
    # Binomial based uncerts sigmaB = sqrt(B/N + B^3/N)
    sigmaB_binomial = np.sqrt(np.array(bRatio_linear_outer)/nManipTot + np.array(bRatio_linear_outer)**3/nManipTot)
    
    # bRatio from linear fit
    plt.figure(2)
    ax = plt.gca()
    ax2 = ax.secondary_yaxis("right", functions=(des_percent, percent_to_bR))
    plt.errorbar(currents, bRatio_linear_outer, yerr = sigmaB_binomial, fmt='ks', markersize=4, capsize=4, capthick=1)
    # plt.errorbar(currents, bRatio_linear_outer, yerr = sigmaB_beta, fmt='bs', markersize=4, capsize=4, capthick=1)
    plt.xlabel('Current (pA)')
    plt.ylabel(r'$k_d$/$k_s$')
    ax2.set_ylabel('Desorbed %')
    plt.text(800, 9, '+1.6 V', fontsize = 12)
    plt.ylim(0, 10)
    plt.xlim(0, 1000)
     # now deal with the ticks
    plt.tick_params(direction='in', length=4,            # A reasonable length
                bottom=True, left=True, top=True, right=False) # want these (inwards on 3 axes) ticks on 3 axes. Want other (outwards) on 2ndary y axis
    # ax = plt.gca()
    xloc = ticker.MultipleLocator(base=100) # this locator puts ticks at regular intervals
    ax.xaxis.set_major_locator(xloc)
    yloc = ticker.MultipleLocator(base=1.0) # this locator puts ticks at regular intervals
    ax.yaxis.set_major_locator(yloc)
    yloc2 = ticker.MultipleLocator(base=5.0) # this locator puts ticks at regular intervals
    ax2.yaxis.set_major_locator(yloc2)
    n = 2  # Keeps every 2nd tick label - use 0 or 1 at end next line for where want start
    [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if i % n != 1]
    [l.set_visible(False) for (i,l) in enumerate(ax.yaxis.get_ticklabels()) if i % n != 1]
    [l.set_visible(False) for (i,l) in enumerate(ax2.yaxis.get_ticklabels()) if i % n != 0]
    plt.savefig('figures/bRatioLinearFitCurrent.png')
    plt.savefig('figures/bRatioLinearFitCurrent.pdf')
    plt.close()
    
    # the rates for each reaction path (part (a) of the figure)
    sigmak_des, sigmak_diff = path_rates_errs(uncert_kdiff_rates_outer, uncert_kdes_rates_outer, k_diff_outer, k_des_outer, sigmaB_binomial, bRatio_linear_outer)
    # and the fit for the n electron process 
    # first the log-log and linear method                                          #[n, ln(B)]
    popt_des, pcov_des = curve_fit(line_c, np.log(currents), np.log(k_des_outer), p0=[1, -10])
    popt_diff, pcov_diff = curve_fit(line_c, np.log(currents), np.log(k_diff_outer), p0=[1, -10])
    popt_Tot, pcov_Tot = curve_fit(line_c, np.log(currents), np.log(K_Tot_outer), p0=[1, -10])
    print(f'B_des: {round_to_uncertainty(popt_des[1], pcov_des[1,1]**0.5)[0]} \u00B1 {round_to_uncertainty(pcov_des[1,1]**0.5, pcov_des[1,1]**0.5)[1]}\n n_des: {round_to_uncertainty(popt_des[0], pcov_des[0,0]**0.5)[0]} \u00B1 {round_to_uncertainty(pcov_des[0,0]**0.5, pcov_des[0,0]**0.5)[1]}\n')
    print(f'B_diff: {round_to_uncertainty(popt_diff[1], pcov_diff[1,1]**0.5)[0]} \u00B1 {round_to_uncertainty(pcov_diff[1,1]**0.5, pcov_diff[1,1]**0.5)[1]}\n n_diff: {round_to_uncertainty(popt_diff[0], pcov_diff[0,0]**0.5)[0]} \u00B1 {round_to_uncertainty(pcov_diff[0,0]**0.5, pcov_diff[0,0]**0.5)[1]}\n')
    print(f'B_Tot: {round_to_uncertainty(popt_Tot[1], pcov_Tot[1,1]**0.5)[0]} \u00B1 {round_to_uncertainty(pcov_Tot[1,1]**0.5, pcov_Tot[1,1]**0.5)[1]}\n n_Tot: {round_to_uncertainty(popt_Tot[0], pcov_Tot[0,0]**0.5)[0]} \u00B1 {round_to_uncertainty(pcov_Tot[0,0]**0.5, pcov_Tot[0,0]**0.5)[1]}\n')
    # then the power law method with weighted fit option available                                            #[B, n]
    # popt_des, pcov_des = curve_fit(power_law, currents, k_des_outer, p0=[0.001, 1])
    # popt_diff, pcov_diff = curve_fit(power_law, currents, k_diff_outer, p0=[0.001, 1])
    # popt_Tot, pcov_Tot = curve_fit(power_law, currents, K_Tot_outer, p0=[0.001, 1])
    popt_des, pcov_des = curve_fit(power_law, currents, k_des_outer, sigma=uncert_kdes_rates_outer, absolute_sigma=False, p0=[0.001, 1])
    popt_diff, pcov_diff = curve_fit(power_law, currents, k_diff_outer, sigma=uncert_kdiff_rates_outer, absolute_sigma=False, p0=[0.001, 1])
    popt_Tot, pcov_Tot = curve_fit(power_law, currents, K_Tot_outer, sigma=uncert_KTot_rates_outer, absolute_sigma=False, p0=[0.001, 1])
    print(f'B_des: {round_to_uncertainty(popt_des[0], pcov_des[0,0]**0.5)[0]} \u00B1 {round_to_uncertainty(pcov_des[0,0]**0.5, pcov_des[0,0]**0.5)[1]}\n n_des: {round_to_uncertainty(popt_des[1], pcov_des[1,1]**0.5)[0]} \u00B1 {round_to_uncertainty(pcov_des[1,1]**0.5, pcov_des[1,1]**0.5)[1]}\n')
    print(f'B_diff: {round_to_uncertainty(popt_diff[0], pcov_diff[0,0]**0.5)[0]} \u00B1 {round_to_uncertainty(pcov_diff[0,0]**0.5, pcov_diff[0,0]**0.5)[1]}\n n_diff: {round_to_uncertainty(popt_diff[1], pcov_diff[1,1]**0.5)[0]} \u00B1 {round_to_uncertainty(pcov_diff[1,1]**0.5, pcov_diff[1,1]**0.5)[1]}\n')
    print(f'B_Tot: {round_to_uncertainty(popt_Tot[0], pcov_Tot[0,0]**0.5)[0]} \u00B1 {round_to_uncertainty(pcov_Tot[0,0]**0.5, pcov_Tot[0,0]**0.5)[1]}\n n_Tot: {round_to_uncertainty(popt_Tot[1], pcov_Tot[1,1]**0.5)[0]} \u00B1 {round_to_uncertainty(pcov_Tot[1,1]**0.5, pcov_Tot[1,1]**0.5)[1]}\n')
    # now get to plotting the rates
    plt.figure(11)
    plt.yscale('log')
    plt.xscale('log')
    plt.errorbar(currents, k_des_outer, yerr = sigmak_des, fmt='ko', markersize=4, capsize=4, capthick=1, label='Desorbed', mfc='white')
    plt.errorbar(currents, k_diff_outer, yerr = sigmak_diff, fmt='ko', markersize=4, capsize=4, capthick=1, label='Switched')
    # add the fit to the plot
    xpoints = np.linspace(min(currents), max(currents), 500)
    plt.plot(xpoints, power_law(xpoints, popt_des[0], popt_des[1]), 'k')
    plt.plot(xpoints, power_law(xpoints, popt_diff[0], popt_diff[1]), 'k')
    plt.xlabel('Injection current (pA)')
    plt.ylabel(r'Rate of manipulation ($s^{-1}$)')
    plt.legend(loc='upper left')
    # now deal with the ticks
    plt.tick_params(direction='in', which='both', length=4,            # A reasonable length
                bottom=True, left=True, top=True, right=True)
    # ax = plt.gca()
    # xloc = ticker.MultipleLocator(base=0.1) # this locator puts ticks at regular intervals
    # ax.xaxis.set_major_locator(xloc)
    # # yloc = ticker.MultipleLocator(base=1.0) # this locator puts ticks at regular intervals
    # # ax.yaxis.set_major_locator(yloc)
    # n = 2  # Keeps every 2nd tick label - use 0 or 1 at end next line for where want start
    # [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if i % n != 0]
    # # [l.set_visible(False) for (i,l) in enumerate(ax.yaxis.get_ticklabels()) if i % n != 1]
    plt.savefig('figures/pathRatesCurrent.png')
    plt.savefig('figures/pathRatesCurrent.pdf')
    plt.close()

    # bRatio from final points
    plt.figure(3)
    ax = plt.gca()
    ax2 = ax.secondary_yaxis("right", functions=(des_percent, percent_to_bR))
    plt.errorbar(currents, bRatio_finalPoint, yerr = sigmaB_beta_finalP, fmt='ks', markersize=4, capsize=4, capthick=1)
    plt.xlabel('Current (pA)')
    plt.ylabel(r'$k_d$/$k_s$')
    ax2.set_ylabel('Desorbed %')
    plt.savefig('figures/bRatioFinalPointCurrent.png')
    plt.close()
    
    # Residual linearFit - finalPoint
    plt.figure(4)
    # ax = plt.gca()
    # ax2 = ax.secondary_yaxis("right", functions=(des_percent, percent_to_bR))
    plt.plot(currents, [bR_lin - bR_fin for bR_lin, bR_fin in zip(bRatio_linear_outer, bRatio_finalPoint)], 'ko')
    plt.xlabel('Current (pA)')
    plt.ylabel('Residual (Linear fit - final point)')
    # ax2.set_ylabel('Desorbed %')
    plt.savefig('figures/residualLinear_FinalPointCurrent.png')
    plt.close()
    
    # Uncorrelated dBF uncert
    # sigmaB = DeltakDes/kDes + DeltakDiff/kDiff
    sigmaB_dBF_uncorrelated = [Ddes + Ddiff for Ddes, Ddiff in zip(uncert_kdes_dbF_outer, uncert_kdiff_dbF_outer)]
    # bRatio from double fit
    plt.figure(5)
    ax = plt.gca()
    ax2 = ax.secondary_yaxis("right", functions=(des_percent, percent_to_bR))
    # plt.plot(currents, bRatio_doubleFit_outer, 'ks')
    plt.errorbar(currents, bRatio_doubleFit_outer, yerr=sigmaB_dBF_uncorrelated, fmt='ks', markersize=4, capsize=4, capthick=1)
    plt.xlabel('Current (pA)')
    plt.ylabel(r'$k_d$/$k_s$')
    ax2.set_ylabel('Desorbed %')
    plt.ylim(1.5, 8.5)
    plt.savefig('figures/bRatioDoubleFitCurrent.png')
    plt.close()
    
    # Residual linearFit - finalPoint
    plt.figure(6)
    # ax = plt.gca()
    # ax2 = ax.secondary_yaxis("right", functions=(des_percent, percent_to_bR))
    plt.plot(currents, [bR_lin - bR_dbF for bR_lin, bR_dbF in zip(bRatio_linear_outer, bRatio_doubleFit_outer)], 'ko')
    plt.xlabel('Current (pA)')
    plt.ylabel('Residual (Linear fit - double fit)')
    # ax2.set_ylabel(r'\Delta Desorbed %')
    plt.savefig('figures/residualLinear_doubleFitCurrent.png')
    plt.close()
    
    plt.figure(fig_GoF_lin)
    fig_GoF_lin.subplots_adjust(wspace=0.5, hspace=0.4) # adjust width spacing so 2nd y axis not overlapping
    plt.tight_layout()
    plt.savefig('figures/GoF_bRatioLinearFitCurrent.png')
    plt.close()
    
    plt.figure(fig_GoF_KTot)
    fig_GoF_lin.subplots_adjust(wspace=0.5, hspace=0.4) # adjust width spacing so 2nd y axis not overlapping
    plt.tight_layout()
    plt.savefig('figures/GoF_KTotCurrent.png')
    plt.close()
    
    plt.figure('fig_beta_pdf')
    plt.xlabel(r'$k_{des}/k_{diff}$')
    plt.ylabel('pdf')
    plt.legend()
    plt.savefig('figures/beta_pdfsCurrent.png')
    plt.close()
    
    return

def branchRatioPosition(positions, df_manip):
    
    print(positions)
    
    # dirty n stayed didct to add to N0
    nStayedList = [24, 127, 58]
    stayedDict = dict(zip(positions, nStayedList))
    
    # Hold total nDes, nDiff and the calculated branching ratios
    nDesTot = []
    nDiffTot = []
    
    bRatio_linear_outer = []
    
    k_des_outer = []
    k_diff_outer = []
    
    # Hold fit uncerts
    uncert_fit_linear = []
    uncert_kdes_rates_outer = []
    uncert_kdiff_rates_outer = []
    
    # Setup colourmap colour for each voltage
    colours = plt.cm.jet(np.linspace(0.0, 0.9, len(positions)))
    
    # Find the rate and BR for each position
    for position, colour in zip(positions, colours):
        print(f'Looking at {position}')
        
        # Set up a mask for the position
        pos_mask = (df_manip['Position']==position)
        df_manip_masked = df_manip[pos_mask]
        
        # Extract final counts of des and diff
        ndestot = df_manip_masked['nDes'].iloc[-1]
        ndifftot = df_manip_masked['nDiff'].iloc[-1]
        N_0 = ndestot + ndifftot + stayedDict[position]
        nDesTot.append(ndestot)
        nDiffTot.append(ndifftot)
        
        # Perform linear fit for BR and add to outer lists
        bR_lin, uncert_fit_lin = bR_linear(df_manip_masked)
        bRatio_linear_outer.append(bR_lin)
        uncert_fit_linear.append(uncert_fit_lin)
        # Plot the data with the fit
        xpoints = np.linspace(0, max(df_manip_masked['nDiff']))
        plt.figure(1)
        plt.plot(df_manip_masked['nDiff'], df_manip_masked['nDes'], 'x', color=colour)
        plt.plot(xpoints, line(xpoints, bR_lin), '--', color=colour)
        
        # Find the rates for each channel needed for part a) of the figure
        k_des, k_diff, K_Tot, uncert_tup_rates = path_rates(df_manip_masked, N_0)
        k_des_outer.append(k_des)
        k_diff_outer.append(k_diff)
        uncert_kdes_rates_outer.append(uncert_tup_rates[0])
        uncert_kdiff_rates_outer.append(uncert_tup_rates[1])
    # end loop over positions
    
    # Calculate total manipulated
    nManipTot = [ndes + ndiff for ndes, ndiff in zip(nDesTot, nDiffTot)]
    
    # Plot the raw counts
    plt.figure(1)
    plt.xlabel('Number of Diffused Molecules')
    plt.ylabel('Number of Desorbed Molecules')
    plt.savefig('figures/nDesDiffRawLinearFitPosition.png')
    plt.close()
    
    # Binomial based uncerts sigmaB = sqrt(B/N + B^3/N)
    sigmaB_binomial = np.sqrt(np.array(bRatio_linear_outer)/nManipTot + np.array(bRatio_linear_outer)**3/nManipTot)

    # convert positins from strings to floats
    positions = [float(pos) for pos in positions]
    
    # bRatio from linear fit
    plt.figure(2)
    plt.errorbar(positions, bRatio_linear_outer, yerr = sigmaB_binomial, fmt='ks', markersize=4, capsize=4, capthick=1)
    plt.text(400, 0.8, '+1.8 V', fontsize = 12)
    ax = plt.gca()
    ax2 = ax.secondary_yaxis("right", functions=(des_percent, percent_to_bR))
    plt.xlabel('Injection position (pm)')
    plt.ylabel(r'$k_d$/$k_s$')
    ax2.set_ylabel('Desorbed %')
    # plt.text(800, 9, '+1.6 V', fontsize = 12)
    plt.ylim(0, 10)
    plt.xlim(-50, 500)
     # now deal with the ticks
    plt.tick_params(direction='in', length=4,            # A reasonable length
                bottom=True, left=True, top=True, right=False) # want these (inwards on 3 axes) ticks on 3 axes. Want other (outwards) on 2ndary y axis
    xloc = ticker.MultipleLocator(base=50) # this locator puts ticks at regular intervals
    ax.xaxis.set_major_locator(xloc)
    yloc = ticker.MultipleLocator(base=1.0) # this locator puts ticks at regular intervals
    ax.yaxis.set_major_locator(yloc)
    yloc2 = ticker.MultipleLocator(base=5.0) # this locator puts ticks at regular intervals
    ax2.yaxis.set_major_locator(yloc2)
    n = 2  # Keeps every 2nd tick label - use 0 or 1 at end next line for where want start
    [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if i % n != 0]
    [l.set_visible(False) for (i,l) in enumerate(ax.yaxis.get_ticklabels()) if i % n != 1]
    [l.set_visible(False) for (i,l) in enumerate(ax2.yaxis.get_ticklabels()) if i % n != 0]
    plt.savefig('figures/bRatioLinearFitPosition.png')
    plt.savefig('figures/bRatioLinearFitPosition.pdf')
    plt.close()
    
    # the rates for each reaction path (part (a) of the figure)
    sigmak_des, sigmak_diff = path_rates_errs(uncert_kdiff_rates_outer, uncert_kdes_rates_outer, k_diff_outer, k_des_outer, sigmaB_binomial, bRatio_linear_outer)
    plt.figure(3)
    # plt.yscale('log')
    plt.errorbar(positions, k_des_outer, yerr = sigmak_des, fmt='ko', markersize=4, capsize=4, capthick=1, label='Desorbed', mfc='white')
    plt.errorbar(positions, k_diff_outer, yerr = sigmak_diff, fmt='ko', markersize=4, capsize=4, capthick=1, label='Switched')
    plt.xlabel('Injection position (pm)')
    plt.ylabel(r'Rate of manipulation ($s^{-1}$)')
    plt.xlim(-50, 500)
     # now deal with the ticks
    plt.tick_params(direction='in', length=4, which='both',           # A reasonable length
                bottom=True, left=True, top=True, right=True) # want these (inwards on 3 axes) ticks on 3 axes. Want other (outwards) on 2ndary y axis
    ax = plt.gca()
    xloc = ticker.MultipleLocator(base=50) # this locator puts ticks at regular intervals
    ax.xaxis.set_major_locator(xloc)
    yloc = ticker.MultipleLocator(base=1) # this locator puts ticks at regular intervals
    ax.yaxis.set_major_locator(yloc)
    yloc2 = ticker.MultipleLocator(base=5.0) # this locator puts ticks at regular intervals
    n = 2  # Keeps every 2nd tick label - use 0 or 1 at end next line for where want start
    [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if i % n != 0]
    # [l.set_visible(False) for (i,l) in enumerate(ax.yaxis.get_ticklabels()) if i % n != 1]
    plt.ylim(0, 4)
    plt.savefig('figures/pathRatesPosition.png')
    plt.savefig('figures/pathRatesPosition.pdf')
    plt.yscale('log')
    plt.ylim(0.05, 5)
    plt.savefig('figures/pathRatesPositionLog.png')
    plt.savefig('figures/pathRatesPositionLog.pdf')
    plt.close()
    
    return

def STS_fitting(ratesTuple):
    #############################################
    # FC -> faulted corner
    # tFM -> toluene faulted middle 
    # tFC -> toluene faulted corner (franken STS)
    #############################################
    
    # extract info found in branchRatioBias from the tuple passed
    # contains the bias (ratesV) at which the total rate (K_Tot) and the uncert (uncert_K_Tot) are found
    ratesV, K_Tot, uncert_K_Tot = ratesTuple
    
    # import the clean atom (clA) STS csv
    df_clAtom = pd.read_csv('data/STS_df.csv')
    # remove the additional index column from the matlab -> csv -> python df
    df_clAtom.drop(columns='Unnamed: 0', inplace=True)
    
    # and the toluene FM STS csv, removing the the extra metadata line and adatom data
    df_tFM = pd.read_csv('data/tFM_STS.csv', skiprows=1, delimiter="\t", usecols=[0, 1, 2])
    
    # set the mask to FC adtaoms and +ve bias
    FCmask = (df_clAtom['site']=='FaultedCorner') & (df_clAtom['Bias / V'] >= 0)
    df_FC = df_clAtom[FCmask]
    
    # fitting gaussian peaks to the FC STS
    # initial parameter guessses for FC
    FC_p0 = [[0.46, 0.14, 1.05], [0.99, 0.36, 2.16], [2.27, 0.27, 2.25], [2.96, 0.32, 1.07], [5.11, 1.3, 9.55]]
    poptFC, pcov = curve_fit(sum_n_gaussians, df_FC['Bias / V'], df_FC['STS'], FC_p0, maxfev=5000)
    print(f"poptFC: {poptFC}")
    
    plt.figure(1)
    # start by plotting the STS
    plt.plot(df_FC['Bias / V'], df_FC['STS'], label='Clean FC STS')
    plt.fill_between(df_FC['Bias / V'], df_FC['STS']-df_FC['error'], df_FC['STS']+df_FC['error'], alpha=0.4)
    # then the total fit
    plt.plot(df_FC['Bias / V'], sum_n_gaussians(df_FC['Bias / V'], *poptFC), '--k', alpha=0.8, label='Total fit')
    # and finally the individual state fits
    for i in range(0, len(poptFC), 3):
        plt.plot(df_FC['Bias / V'], gaussian(df_FC['Bias / V'], *poptFC[i:i+3]), '--k', alpha=0.4, label='Contributing fits')
    plt.ylabel(r'(dI/dV)/(I/V)')
    plt.xlabel('Bias (V)')
    fontP = FontProperties() # making legend smaller
    fontP.set_size('small')
    # and removing the duplicate labels
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper left', prop=fontP)
    # now deal with the ticks
    plt.tick_params(direction='in', length=4,            # A reasonable length
                bottom=True, left=True, top=True, right=True)
    ax = plt.gca()
    xloc = ticker.MultipleLocator(base=0.5) # this locator puts ticks at regular intervals
    ax.xaxis.set_major_locator(xloc)
    yloc = ticker.MultipleLocator(base=0.5) # this locator puts ticks at regular intervals
    ax.yaxis.set_major_locator(yloc)
    n = 2  # Keeps every 2nd tick label - use 0 or 1 at end next line for where want start
    # [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if i % n != 1]
    [l.set_visible(False) for (i,l) in enumerate(ax.yaxis.get_ticklabels()) if i % n != 1]
    plt.xlim(0, 2.25)
    plt.ylim(top=4)
    plt.savefig('figures/cleanFC_STS.png')
    plt.savefig('figures/cleanFC_STS.pdf')
    plt.close()
    
    # define the integration points out here to ensure they are consistent across each call to the integration function
    nIntPoints = 5000
    integrationVpoints = np.linspace(0, max(df_FC['Bias / V']), nIntPoints)
    # Get and plot the fraction captured in each clean FC state found above
    FCcapturedFracArray = frac_I_captured(integrationVpoints, poptFC)
    plt.figure(3)
    for state in FCcapturedFracArray:
        plt.plot(integrationVpoints, state)
    plt.xlabel('Sample bias (V)')
    plt.ylabel('$s_i(V)$')
    plt.xlim(0, 2.25)
    plt.savefig('figures/cleanFC_sV.png')
    plt.savefig('figures/cleanFC_sV.pdf')
    plt.close()
    
    # Fit total rates to the frac captured with a prefactor
    # Prefer fit in log space as error bars are similar magnitude. Keep linear in for comparision. Could weight fit in future 
    eachStateFracCapAtRateV = match_V_to_frac_captured(integrationVpoints, ratesV, FCcapturedFracArray)
    FC_rates_p0 = [1e-6, 1e-10, 1223, 0.001, 0.001]
    FC_rates_bounds = ([0, 0, 0, 0, 0], [10,10,5000,5000,0.01])
    beta_param_FC = partial(beta_param_log, fixed_fracCapturedArray=eachStateFracCapAtRateV)
    Logpopt, Logpcov = curve_fit(beta_param_FC, ratesV, np.log10(K_Tot), p0=FC_rates_p0, bounds=FC_rates_bounds)
    
    beta_param_FC_lin = partial(beta_param_lin, fixed_fracCapturedArray=eachStateFracCapAtRateV)
    Linpopt, Linpcov = curve_fit(beta_param_FC_lin, ratesV, K_Tot, p0=FC_rates_p0, bounds=FC_rates_bounds)
    
    print(f'Logpopt FC: {Logpopt}')
    print(f'Linpopt FC: {Linpopt}')
    
    plt.figure(6)
    plt.errorbar(ratesV, K_Tot, yerr=uncert_K_Tot, fmt='ko', markersize=4, capsize=4, capthick=1)
    # plot the fit for the total rate
    plt.plot(integrationVpoints, 10**beta_param_log(integrationVpoints, *Logpopt, fixed_fracCapturedArray=FCcapturedFracArray), '-k')
    # plt.plot(integrationVpoints, beta_param_lin(integrationVpoints, *Linpopt, fixed_fracCapturedArray=FCcapturedFracArray), '-k')
    # and the fitted amplitude of each state
    for beta, state in zip(Logpopt, FCcapturedFracArray):
        plt.plot(integrationVpoints, beta*state, '--')
    # for beta, state in zip(Linpopt, FCcapturedFracArray):
    #     plt.plot(integrationVpoints, beta*state, '--')
    plt.xlabel('Sample Bias / V') 
    plt.ylabel('Total rate $K$')
    ax = plt.gca()
    ax2 = ax.secondary_yaxis("right")
    ax2.set_ylabel('$\\sum  k_i s_i(V) $')
    plt.xlim(1.3, 2.25)
    plt.savefig('figures/FC_betaFit.png')
    plt.yscale('log')
    plt.ylim(1e-2, 10)
    plt.savefig('figures/FC_betaFitLog.png')
    plt.close()
    
    # now fit peaks to the tFM STS
    # initial parameter guessses for tFM
    tFM_p0 = [[0.43, 0.17, 1.3], [1.18, 0.28, 2.3], [1.7, 0.23, 2.7], [2.26, 0.25, 3.1]]
    tFM_bounds_low = [0.97*p for p0 in tFM_p0 for p in p0]
    tFM_bounds_upp = [1.15*p for p0 in tFM_p0 for p in p0]
    tFM_bounds = (tFM_bounds_low, tFM_bounds_upp)
    
    # had to flatten the 2D p0 array as bounds cannot be accpeted with `*` so both must be 1D
    popttFM, pcov = curve_fit(sum_n_gaussians, df_tFM['Sample bias (V)'], df_tFM['(dI/dV)/(I/V)'], np.array(tFM_p0).flatten(), bounds=tFM_bounds, maxfev=5000)
    print(f"popttFM: {popttFM}")
    
    plt.figure(2)
    # again start with the STS
    plt.plot(df_tFM['Sample bias (V)'], df_tFM['(dI/dV)/(I/V)'], label='tFM STS')
    plt.fill_between(df_tFM['Sample bias (V)'], df_tFM['(dI/dV)/(I/V)']-df_tFM['Error'], df_tFM['(dI/dV)/(I/V)']+df_tFM['Error'], alpha=0.4)
    # then the total fit
    plt.plot(df_tFM['Sample bias (V)'], sum_n_gaussians(df_tFM['Sample bias (V)'], *popttFM), '--k', alpha=0.8, label='Total fit')
    # and the individual state fits
    fitXpoints = np.linspace(0, 2.25, 1000) # need these as tFM STS only out to 2V but need range out to 2.25 to cover the bR range
    for i in range(0, len(popttFM), 3):
        plt.plot(fitXpoints, gaussian(fitXpoints, *popttFM[i:i+3]), '--k', alpha=0.4, label='Contributing fits')
    plt.ylabel(r'(dI/dV)/(I/V)')
    plt.xlabel('Bias (V)')
    fontP = FontProperties() # making legend smaller
    fontP.set_size('small')
    # and removing the duplicate labels
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper left', prop=fontP)
    # now deal with the ticks
    plt.tick_params(direction='in', length=4,            # A reasonable length
                bottom=True, left=True, top=True, right=True)
    plt.xlim(0, 2.25)
    plt.savefig('figures/tFM_STS.png')
    plt.savefig('figures/tFM_STS.pdf')
    
    # Get and plot the fraction captured in each tFM state found above
    tFMcapturedFracArray = frac_I_captured(integrationVpoints, popttFM)
    plt.figure(4)
    for state in tFMcapturedFracArray:
        plt.plot(integrationVpoints, state)
    plt.xlabel('Sample bias (V)')
    plt.ylabel('$s_i(V)$')
    plt.xlim(0, 2.25)
    plt.savefig('figures/tFM_sV.png')
    plt.savefig('figures/tFM_sV.pdf')
    plt.close()
    
    # Fit total rates to the frac captured with a prefactor
    # Prefer fit in log space as error bars are similar magnitude. Keep linear in for comparision. Could weight fit in future 
    eachStateFracCapAtRateV = match_V_to_frac_captured(integrationVpoints, ratesV, tFMcapturedFracArray)
    tFM_rates_p0 = [1.00000000e-10, 1.44668024e-05, 50, 1]
    tFM_rates_bounds = ([0, 0, 0, 0], [0.003,0.0001,100,100])
    beta_param_tFM = partial(beta_param_log, fixed_fracCapturedArray=eachStateFracCapAtRateV)
    Logpopt, Logpcov = curve_fit(beta_param_tFM, ratesV, np.log10(K_Tot), p0=tFM_rates_p0, bounds=tFM_rates_bounds)
    
    beta_param_tFM_lin = partial(beta_param_lin, fixed_fracCapturedArray=eachStateFracCapAtRateV)
    Linpopt, Linpcov = curve_fit(beta_param_tFM_lin, ratesV, K_Tot, p0=tFM_rates_p0, bounds=tFM_rates_bounds)
    
    print(f'Logpopt: {Logpopt}')
    print(f'Linpopt: {Linpopt}')
    
    plt.figure(5)
    plt.errorbar(ratesV, K_Tot, yerr=uncert_K_Tot, fmt='ko', markersize=4, capsize=4, capthick=1)
    # plot the fit for the total rate
    plt.plot(integrationVpoints, 10**beta_param_log(integrationVpoints, *Logpopt, fixed_fracCapturedArray=tFMcapturedFracArray), '-k')
    # plt.plot(integrationVpoints, beta_param_lin(integrationVpoints, *Linpopt, fixed_fracCapturedArray=tFMcapturedFracArray), '-k')
    # and the fitted amplitude of each state
    for beta, state in zip(Logpopt, tFMcapturedFracArray):
        plt.plot(integrationVpoints, beta*state, '--')
    # for beta, state in zip(Linpopt, tFMcapturedFracArray):
    #     plt.plot(integrationVpoints, beta*state, '--')
    plt.xlabel('Sample Bias / V') 
    plt.ylabel('Total rate $K$')
    ax = plt.gca()
    ax2 = ax.secondary_yaxis("right")
    ax2.set_ylabel('$\\sum  k_i s_i(V) $')
    plt.xlim(1.3, 2.25)
    # plt.ylim(-2,1)
    plt.savefig('figures/tFM_betaFit.png')
    plt.yscale('log')
    plt.ylim(1e-2, 10)
    plt.savefig('figures/tFM_betaFitLog.png')
    plt.close()
    
    # plot both the cleanFC and the tFM STS on the same figure with an offset applied
    plt.figure(11, figsize=(8,8))
    # start with the tFM STS
    plt.plot(df_tFM['Sample bias (V)'], df_tFM['(dI/dV)/(I/V)'], color='tab:blue', label='Measured STS',)
    plt.fill_between(df_tFM['Sample bias (V)'], df_tFM['(dI/dV)/(I/V)']-df_tFM['Error'], df_tFM['(dI/dV)/(I/V)']+df_tFM['Error'], color='tab:blue', alpha=0.4)
    # then the total fit
    plt.plot(df_tFM['Sample bias (V)'], sum_n_gaussians(df_tFM['Sample bias (V)'], *popttFM), '--k', dashes=(4, 5), alpha=0.8, label='Total fit')
    # and the individual state fits
    fitXpoints = np.linspace(0, 2.25, 1000) # need these as tFM STS only out to 2V but need range out to 2.25 to cover the bR range
    for i in range(0, len(popttFM), 3):
        plt.plot(fitXpoints, gaussian(fitXpoints, *popttFM[i:i+3]), '--', color=state_colour(popttFM[i]), label='Contributing fits')
    # add the cleanFC STS with a constant offset
    plt.plot(df_FC['Bias / V'], df_FC['STS']+4, color='tab:blue', label='Measured STS')
    plt.fill_between(df_FC['Bias / V'], df_FC['STS']+4-df_FC['error'], df_FC['STS']+4+df_FC['error'], color='tab:blue', alpha=0.4)
    # then the total fit with offset
    plt.plot(df_FC['Bias / V'], sum_n_gaussians(df_FC['Bias / V'], *poptFC)+4, '--k', alpha=0.8, dashes=(4, 5), label='Total fit')
    # and finally the individual state fits with offset
    for i in range(0, len(poptFC), 3):
        plt.plot(df_FC['Bias / V'], gaussian(df_FC['Bias / V'], *poptFC[i:i+3])+4, '--', color=state_colour(poptFC[i]), label='Contributing fits')
    plt.ylabel(r'(dI/dV)/(I/V)')
    plt.xlabel('Bias (V)')
    fontP = FontProperties() # making legend smaller
    fontP.set_size('small')
    # and removing the duplicate labels
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper left', prop=fontP)
    # now deal with the ticks
    plt.tick_params(direction='in', length=4,            # A reasonable length
                bottom=True, left=True, top=True, right=True)
    ax = plt.gca()
    xloc = ticker.MultipleLocator(base=0.5) # this locator puts ticks at regular intervals
    ax.xaxis.set_major_locator(xloc)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%g')) # this removes unnecessary trailing decimal points
    yloc = ticker.MultipleLocator(base=1) # this locator puts ticks at regular intervals
    ax.yaxis.set_major_locator(yloc)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))
    n = 2  # Keeps every 2nd tick label - use 0 or 1 at end next line for where want start
    # [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if i % n != 1]
    # [l.set_visible(False) for (i,l) in enumerate(ax.yaxis.get_ticklabels()) if i % n != 1]
    plt.xlim(0, 2.25)
    plt.ylim(top=8)
    plt.vlines([1.4, 1.95], -99, 99, linestyles='--', colors='k', alpha=0.8)
    plt.savefig('figures/tFM-cleanFC_STS.png')
    plt.savefig('figures/tFM-cleanFC_STS.pdf')
    
    # Now combine the LUMO state from the tFM fit and add this into the cleanFC state params and fit with this -> franken STS
    popttFC = np.concatenate((poptFC[:6], popttFM[6:9], poptFC[6:]))
    print(f'popttFC: {popttFC}')
    
    # Plot this new total fit for the combined STS
    plt.figure(77, figsize=(8, 4))
    plt.plot(fitXpoints, sum_n_gaussians(fitXpoints, *popttFC), '--k', dashes=(4, 5), alpha=0.8, label='Total fit')
    # and the individual state fits
    for i in range(0, len(popttFC), 3):
        plt.plot(fitXpoints, gaussian(fitXpoints, *popttFC[i:i+3]), '--', alpha=1, label='Contributing fits', color=state_colour(popttFC[i]))
    plt.ylabel(r'(dI/dV)/(I/V)')
    plt.xlabel('Bias (V)')
    fontP = FontProperties() # making legend smaller
    fontP.set_size('small')
    # and removing the duplicate labels
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper left', prop=fontP)
    # now deal with the ticks
    plt.tick_params(direction='in', length=4,            # A reasonable length
                bottom=True, left=True, top=True, right=True)
    ax = plt.gca()
    xloc = ticker.MultipleLocator(base=0.5) # this locator puts ticks at regular intervals
    ax.xaxis.set_major_locator(xloc)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%g')) # this removes unnecessary trailing decimal points
    yloc = ticker.MultipleLocator(base=0.5) # this locator puts ticks at regular intervals
    ax.yaxis.set_major_locator(yloc)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))
    n = 2  # Keeps every 2nd tick label - use 0 or 1 at end next line for where want start
    # [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if i % n != 1]
    [l.set_visible(False) for (i,l) in enumerate(ax.yaxis.get_ticklabels()) if i % n != 1]
    plt.vlines([1.4, 1.95], -99, 99, linestyles='--', colors='k', alpha=0.8)
    plt.xlim(0, 2.25)
    plt.ylim(-0.2, 4)
    plt.savefig('figures/tFC_STS.png')
    plt.savefig('figures/tFC_STS.pdf')
    plt.close()
    
    # Get and plot the fraction captured in each tFC state found above
    tFCcapturedFracArray = frac_I_captured(integrationVpoints, popttFC)
    plt.figure(8, figsize=(8,4))
    for state, stateE in zip(tFCcapturedFracArray, popttFC[::3]): # slicing with only the step specified
        plt.plot(integrationVpoints, state, color=state_colour(stateE))
    plt.xlabel('Sample bias (V)')
    plt.ylabel('$s_i(V)$')
    plt.xlim(0, 2.25)
    plt.ylim(top=1)
    # now deal with the ticks
    plt.tick_params(direction='in', length=4,            # A reasonable length
                bottom=True, left=True, top=True, right=True)
    ax = plt.gca()
    xloc = ticker.MultipleLocator(base=0.5) # this locator puts ticks at regular intervals
    ax.xaxis.set_major_locator(xloc)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%g')) # this removes unnecessary trailing decimal points
    yloc = ticker.MultipleLocator(base=0.1) # this locator puts ticks at regular intervals
    ax.yaxis.set_major_locator(yloc)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))
    n = 2  # Keeps every 2nd tick label - use 0 or 1 at end next line for where want start
    # [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if i % n != 1]
    [l.set_visible(False) for (i,l) in enumerate(ax.yaxis.get_ticklabels()) if i % n != 1]
    plt.vlines([1.4, 1.95], -99, 99, linestyles='--', colors='k', alpha=0.8)
    plt.savefig('figures/tFC_sV.png')
    plt.savefig('figures/tFC_sV.pdf')
    plt.close()
    
    # Fit total rates to the frac captured with a prefactor
    # Prefer fit in log space as error bars are similar magnitude. Keep linear in for comparision. Could weight fit in future 
    eachStateFracCapAtRateV = match_V_to_frac_captured(integrationVpoints, ratesV, tFCcapturedFracArray)
    
    tFC_rates_p0 = [1e-6, 1e-10, 1000, 1223, 0.00001, 0.00001]
    tFC_rates_bounds = ([0, 0, 0, 0, 0, 0], [10,10,5000,5000,0.0001,0.0001]) 
    
    beta_param_tFC = partial(beta_param_log, fixed_fracCapturedArray=eachStateFracCapAtRateV)
    Logpopt, Logpcov = curve_fit(beta_param_tFC, ratesV, np.log10(K_Tot), p0=tFC_rates_p0, bounds=tFC_rates_bounds)
    
    beta_param_tFC_lin = partial(beta_param_lin, fixed_fracCapturedArray=eachStateFracCapAtRateV)
    Linpopt, Linpcov = curve_fit(beta_param_tFC_lin, ratesV, K_Tot, p0=tFC_rates_p0, bounds=tFC_rates_bounds)
    
    print(f'Logpopt tFC: {Logpopt}')
    print(f'Logpcov: {np.sqrt(np.diag(Logpcov))}')
    print(f'Logpcov LUMO: {Logpcov[2][2]**0.5}')
    print(f'Logpcov U2: {Logpcov[3][3]**0.5}')
    print(f'Linpopt tFC: {Linpopt}')
    
    plt.figure(9, figsize=(8, 4))
    plt.errorbar(ratesV, K_Tot, yerr=uncert_K_Tot, fmt='ko', markersize=4, capsize=4, capthick=1)
    # plt.plot(integrationVpoints, beta_param_lin(integrationVpoints, *Linpopt, fixed_fracCapturedArray=tFCcapturedFracArray), '-k')
    # plot the fitted amplitude of each state
    for beta, state, stateE in zip(Logpopt, tFCcapturedFracArray, popttFC[::3]): 
        plt.plot(integrationVpoints, beta*state, '--', color=state_colour(stateE))
    # and the fit for the total rate
    plt.plot(integrationVpoints, 10**beta_param_log(integrationVpoints, *Logpopt, fixed_fracCapturedArray=tFCcapturedFracArray), '-k')
    # for beta, state in zip(Linpopt, tFCcapturedFracArray):
    #     plt.plot(integrationVpoints, beta*state, '--')
    plt.xlabel('Sample Bias / V')  
    plt.ylabel('Total rate $K$')
    ax = plt.gca()
    ax2 = ax.secondary_yaxis("right")
    ax2.set_ylabel('$\\sum  k_i s_i(V) $')
    plt.xlim(0, 2.25)
    # now deal with the ticks
    plt.tick_params(direction='in', length=4, which='both',      # A reasonable length
                bottom=True, left=True, top=True, right=True)
    ax2.tick_params(right=False, which='both')
    xloc = ticker.MultipleLocator(base=0.5) # this locator puts ticks at regular intervals
    ax.xaxis.set_major_locator(xloc)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%g')) # this removes unnecessary trailing decimal points
    # yloc = ticker.MultipleLocator(base=0.1) # this locator puts ticks at regular intervals
    # ax.yaxis.set_major_locator(yloc)
    # ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))
    n = 2  # Keeps every 2nd tick label - use 0 or 1 at end next line for where want start
    # [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if i % n != 1]
    # [l.set_visible(False) for (i,l) in enumerate(ax.yaxis.get_ticklabels()) if i % n != 1]
    plt.vlines([1.4, 1.95], -99, 99, linestyles='--', colors='k', alpha=0.8)
    plt.ylim(-0.2, 5)
    plt.savefig('figures/tFC_betaFit.png')
    plt.savefig('figures/tFC_betaFit.pdf')
    plt.yscale('log')
    plt.ylim(5e-2, 5)
    plt.savefig('figures/tFC_betaFitLog.png')
    plt.savefig('figures/tFC_betaFitLog.pdf')
    plt.close()
    
    return

def main():
    
    # Supply desorbed then diffused file path
    voltages, df_manipulated = extract_data('data/Desorbed_Data.csv', 'data/Diffused_data.csv', 'Voltage')
    ratesTuple = branchRatioBias(voltages, df_manipulated)
    currents, df_manipulated = extract_data('data/TolueneCurrents_Desorbed.csv', 'data/TolueneCurrents_Diffused.csv', 'Current')
    branchRatioCurrent(currents, df_manipulated)
    positions, df_manipulated = extract_data('data/positions_desorbed.csv', 'data/positions_diffused.csv', 'Position')
    branchRatioPosition(positions, df_manipulated)
    STS_fitting(ratesTuple) # requires the bRBias function to produce the ratesTuple

if __name__=='__main__':
 	main()
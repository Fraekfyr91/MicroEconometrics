import numpy as np 
from scipy.stats import norm

name = 'Tobit'

def q(theta, y, x): 
    return None # Fill in 

def loglikelihood(theta, y, x): 
    assert True # FILL IN: add some assertions to make sure that dimensions are as you assume 

    # unpack parameters 
    b = None # first K parameters are betas, the last is sigma 
    sig = None # take abs() to ensure positivity (in case the optimizer decides to try it)

    xb_s = None
    Phi = None #Cumulative dist.
    
    phi = None #Density
    
    Phi = np.clip(Phi, 1e-8, 1.-1e-8)
    
    ll = None # fill in 

    return ll

def starting_values(y,x): 
    '''starting_values
    Returns
        theta: K+1 array, where theta[:K] are betas, and theta[-1] is sigma (not squared)
    '''
    N,K = x.shape
    
    b_ols = None # OLS estimates as starting values 
    sigmahat = None # OLS estimate of sigma = sqrt(sigma^2) as starting value 
    theta0 = np.append(b_ols, sigmahat)
    return theta0 

def predict(theta, x): 
    '''predict(): the expected value of y given x 
    Returns E, E_pos
        E: E(y|x)
        E_pos: E(y|x, y>0) 
    '''
    b = None
    sig = None
    
    
    # Fill in 
    E = None
    Epos = None
    return E, Epos

def sim_data(theta, N:int): 
    b = None
    sig = None
    K = b.size
    
    #Create x by a Nx1 constant term and a NxK-1 matrix of covariates that are random normally distributed
    x = None 
    
    
    #Create the error term
    u = None
    
    # FILL IN 
    ystar = None
    y = None 

    return y,x

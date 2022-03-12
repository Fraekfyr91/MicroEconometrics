import numpy as np 
from scipy.stats import norm

name = 'Tobit_Het'

def q(theta, y, x, h): 
    return -loglikelihood(theta, y, x, h)

def loglikelihood(
        theta: list, 
        y: np.ndarray, 
        x: np.ndarray, 
        h #Anonymous function for our heteroskedasticity 
    ): 
    
    assert y.ndim == 1, f'y should be 1-dimensional'
    assert theta.ndim == 1, f'theta should be 1-dimensional'

    # unpack parameters 
    N,K = x.shape
    b = theta[:-1] # first K parameters are betas, the last is sigma 
    gamma = np.abs(theta[-1]) # take abs() to ensure positivity 
    N,K = x.shape

    xb = x@b
    sigma_i = gamma*h(xb)
    xb_s = xb/sigma_i
    Phi = norm.cdf(xb_s)

    u_s = (y - x@b)/sigma_i
    phi = norm.pdf(u_s)/sigma_i

    # avoid taking log of zero
    Phi = np.clip(Phi, 1e-8, 1.-1e-8)

    # loglikelihood function 
    ll = (y == 0.0) * np.log(1.0-Phi) + (y > 0) * np.log(phi)

    return ll

def starting_values(y,x): 
    
    N,K = x.shape 
    b_ols = np.linalg.solve(x.T@x, x.T@y)
    res = y - x@b_ols 
    sig2hat = 1./(N-K) * np.dot(res, res)
    sighat = np.sqrt(sig2hat) # our convention is that we estimate sigma, not sigma squared
    theta0 = np.append(b_ols, sighat)
    return theta0 

import numpy as np

# TODO add truncation to all distributions

def gaussian(x,a,mu,sigma):
    return a*np.exp(-(x-mu)**2/(2*sigma**2))

def gaussian_jacobian(x, a, mu, sigma):
    d_da = np.exp(-(x-mu)**2/(2*sigma**2))
    d_dmu = a*((x-mu)/sigma**2)*np.exp(-(x-mu)**2/(2*sigma**2))
    d_dsigma = a*((x-mu)/sigma**3)*np.exp(-(x-mu)**2/(2*sigma**2))
    return np.array([d_da, d_dmu, d_dsigma])

def gaussian_d2_dx2(x, a, mu, sigma):
    return (-a/sigma**2 + a*(x-mu)**2/sigma**4)*np.exp(-(x-mu)**2/(2*sigma**2))

def uniform(x, mean):
    return mean

def uniform_d2_dx2(x, mean):
    return 0

def log_normal(x, a, mu, sigma):
    eps = 1e-14
    return a/((x+eps)*sigma*np.sqrt(2*np.pi))*np.exp(-(np.log(x)-mu)**2/(2*sigma**2))

def log_normal_d2_dx2(x, a, mu, sigma):
    # make x non zero
    if x <1e-14:
        x = 1e-14
    return (np.sqrt(2/np.pi)*a*np.exp(-(np.log(x) - mu)**2/(2*sigma**2)))/(sigma*x**3) + \
        (np.sqrt(2/np.pi)*a*(np.log(x) - mu) *np.exp(-(np.log(x) - mu)**2/(2*sigma**2)))/(sigma**3*x**3) + \
        (a*(((np.log(x) - mu)*np.exp(-(np.log(x) - mu)**2/(2*sigma**2)))/(sigma**2*x**2) - \
        np.exp(-(np.log(x) - mu)**2/(2*sigma**2))/(sigma**2*x**2) + \
        ((np.log(x) - mu)**2 *np.exp(-(np.log(x) - mu)**2/(2*sigma**2)))/(sigma**4*x**2)))/(np.sqrt(2*np.pi)*sigma*x)

#Python3 Steven
#generate proability distrubutions for simulation
#https://docs.scipy.org/doc/scipy/reference/stats.html

import numpy as np
from scipy import stats


def genUniform_distribution(size,start,width): #uniform distribution
    #return stats.uniform.rvs(size=size, loc = start, scale=width)
    return np.random.uniform(size=size, low = start, high=start+width)

def genNormal_distribution(size,u=0,scale=1): #normal distribution
    #return stats.norm.rvs(size=size, loc=u,scale=scale)
    return np.random.normal(size=size,loc=u,scale=scale)

def genGamma_distribution(size,a=5): #gamma distribution
    return stats.gamma.rvs(size=size, a=a)

def genExp_distribution(size,u=0,scale=1): #exponential  distribution
    return stats.expon.rvs(size=size,scale=scale,loc=u)

def genPoisson_distribution(size,mu=3): #poisson  distribution
    return stats.poisson.rvs(size=size,mu=mu)

def genBinomial_distribution(size,n=10,p=0.8): #binomial   distribution
    return stats.binom.rvs(size=size,n=n,p=p)

def genBernoulli_distribution(size,p=0.6): #bernoulli   distribution
    return stats.bernoulli.rvs(size=size,p=p)

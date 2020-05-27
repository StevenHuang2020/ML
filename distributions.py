#Python3 Steven
#common proability distrubutions
import numpy as np
import matplotlib.pyplot as plt
from sympy import integrate
import math
from scipy.special import gamma,beta,factorial
#https://en.wikipedia.org/wiki/Gamma_function
#https://en.wikipedia.org/wiki/Beta_function
#https://en.wikipedia.org/wiki/Factorial
from permutationAndCom import permut,combinat


#---------------------------Discrete distribution------------------------------#
def Discrete_uniform_distribution(x,N=5):#https://en.wikipedia.org/wiki/Discrete_uniform_distribution
    return np.zeros(len(x)) + 1/N

def Binomial_distribution(N,p): #https://en.wikipedia.org/wiki/Binomial_distribution
    assert(N>=1 and 0<=p<=1)
    def Binomial(k):
        return combinat(N,k)*np.power(p,k)*np.power(1-p,N-k)
    
    return list(map(Binomial, [i for i in range(N)]))
    
def Geometric_distribution(N,p):#https://en.wikipedia.org/wiki/Geometric_distribution
    def Geometric(k):
        return np.power(1-p,k)*p
    
    return list(map(Geometric, [i for i in range(N)]))

def Hypergeometric_distribution(N,K,n): #https://en.wikipedia.org/wiki/Hypergeometric_distribution
    def Hypergeometric(k):
        return combinat(K,k)*combinat(N-K,n-k)/combinat(N,n)
    
    return list(map(Hypergeometric, [i for i in range(n)]))

def ZipfsLaw(N=10, s=1):#https://en.wikipedia.org/wiki/Zipf%27s_law
    v = np.sum(1/np.power([i+1 for i in range(N)],s))
    def ZipfLaw(k):
        return (1/np.power(k,s))/v
    
    return list(map(ZipfLaw, [i+1 for i in range(N)]))

#---------------------------Continuous distribution------------------------------#
def Uniform_distribution(x,a=1,b=3): #https://en.wikipedia.org/wiki/Uniform_distribution_(continuous)
    return np.zeros(len(x)) + 1/(b-a)

def NormalDistribution_pdf(x, delta=1, u=0): #https://en.wikipedia.org/wiki/Normal_distribution
    return (1/delta*np.sqrt(2*np.pi))*np.exp(-0.5*((x-u)/delta)**2)
  
def Cauchy_pdf(x, x0=0, scaler=1):#https://en.wikipedia.org/wiki/Cauchy_distribution
    return (1/(np.pi*scaler))*(scaler**2/((x-x0)**2+scaler**2))
  
def Generalized_logistic_distribution(x,alpha=1):#https://en.wikipedia.org/wiki/Generalized_logistic_distribution
    return alpha*np.exp(-1*x)/np.power(1+np.exp(-1*x),alpha+1)

def Gumbel_distribution(x,u=0,belta=1):#https://en.wikipedia.org/wiki/Gumbel_distribution
    z = (x-u)/belta
    return np.exp(-1*(z + np.exp(-1*z)))/belta

def Laplace_distribution(x,u=0,b=1):#https://en.wikipedia.org/wiki/Laplace_distribution
    return np.exp(-1*np.abs(x-u)/b)
   
def Logistic_distribution(x, u=0, s=1):#https://en.wikipedia.org/wiki/Logistic_distribution
    z = np.exp(-1*(x-u)/s)
    return z/(s*(1+z)**2)

def Gamma_distribution(x,k=1,theta=2): #https://en.wikipedia.org/wiki/Gamma_distribution
    return np.power(x,k-1)*np.exp(-1*x/theta)/(gamma(k)*np.power(theta,k))

def StudentT_distribution(x,v=1):#https://en.wikipedia.org/wiki/Student%27s_t-distribution
    z = np.power(1+x**2/v, -0.5*(v+1))
    return z*gamma((v+1)/2)/(gamma(v/2)*np.sqrt(np.pi*v))
    #return z/(beta(0.5, v/2)*np.sqrt(v))

def Log_normal_distribution(x, delta=1, u=0):#https://en.wikipedia.org/wiki/Log-normal_distribution
    z = -1*(np.log(x)-u)**2/(2*delta**2)
    return (1/(x*delta*np.sqrt(2*np.pi)))* np.exp(z)

def Weibull_distribution(x,lamda=1,k=1): #https://en.wikipedia.org/wiki/Weibull_distribution
    y = np.zeros((len(x),))
    l = np.where(x < 0)
    if len(l) != 0:
        y[l[0]]=0
        
    l = np.where(x >= 0)
    if len(l) != 0:
        y[l[0]] = (k/lamda)*np.power(x[l[0]]/lamda,k-1)*np.exp(-1*np.power(x[l[0]]/lamda,k))
    return y

def Pareto_distribution(x,alpha=1,Xm=1):#https://en.wikipedia.org/wiki/Pareto_distribution
    y = np.zeros((len(x),))
    l = np.where(x < Xm)
    if len(l) != 0:
        y[l[0]]=0
    
    l = np.where(x >= Xm)
    if len(l) != 0:
        y[l[0]] = alpha*np.power(Xm,alpha)/np.power(x[l[0]],alpha+1)
    return y

def Rayleigh_distribution(x,delta=0.5):#https://en.wikipedia.org/wiki/Rayleigh_distribution
    y = np.zeros((len(x),))
    l = np.where(x >= 0)
    if len(l) != 0:
        y[l[0]] = (x/delta**2)*np.exp(-1*x**2/(2*delta**2))
    return y
    
def Beta_distribution(x,alpha=0.5,ba=0.5): #https://en.wikipedia.org/wiki/Beta_distribution
    return np.power(x, alpha-1)*np.power(1-x, ba-1)/beta(alpha,ba)
 
 
#------------------------------------main----------------------------------#   
def main():
    pass 
    
if __name__ == '__main__':
    main()

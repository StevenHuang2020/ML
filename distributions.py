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

def Poisson_distribution(N=20, lam=1): #https://en.wikipedia.org/wiki/Poisson_distribution
    def Poisson(k):
        return np.power(lam,k)*np.power(np.e, -1*lam)/factorial(k)
    
    return list(map(Poisson, [i for i in range(N)]))

def ZipfsLaw(N=10, s=1):#https://en.wikipedia.org/wiki/Zipf%27s_law
    v = np.sum(1/np.power([i+1 for i in range(N)],s))
    def ZipfLaw(k):
        return (1/np.power(k,s))/v
    
    return list(map(ZipfLaw, [i+1 for i in range(N)]))

def Beta_binomial_distribution(N=10,alpha=0.2,bta=0.25): #https://en.wikipedia.org/wiki/Beta-binomial_distribution
    def Beta_binomial(k):
        return combinat(N,k)*beta(k+alpha, N-k+bta)/beta(alpha,bta)
    
    return list(map(Beta_binomial, [i for i in range(N)]))

def Logarithmic_distribution(N,p=0.33): #https://en.wikipedia.org/wiki/Logarithmic_distribution
    def Logarithmic(k):
        return -1*np.power(p,k)/(np.log(1-p)*k)
    
    return list(map(Logarithmic, [i+1 for i in range(N)]))

def Conway_Maxwell_Poisson_distribution(N=20,lam=1, v=1.5): # https://en.wikipedia.org/wiki/Conway%E2%80%93Maxwell%E2%80%93Poisson_distribution
    def Conway(k):
        return np.power(lam,k)/np.power(factorial(k),v)
    
    max = 10000
    z = np.sum(list(map(Conway, [i for i in range(max)])))
    res = list(map(Conway, [i for i in range(N)]))
    return res/z

def Modified_Bessel_functions(x,alpha=0): #https://en.wikipedia.org/wiki/Bessel_function#Modified_Bessel_functions
    def Bessel(k):
        return 1/(factorial(k)*gamma(k+alpha+1))*np.power(0.2*x,2*k+alpha)
    
    max = 1000
    return np.sum(list(map(Bessel, [i for i in range(max)])))

def Skellam_distribution(N=8,u1=1,u2=1): #https://en.wikipedia.org/wiki/Skellam_distribution
    def Skellam(k):
        return np.exp(-u1-u2)*np.power(u1/u2, 0.5*k)*Modified_Bessel_functions(x=2*np.sqrt(u1*u2),alpha=k)
    return list(map(Skellam, [i-N/2 for i in range(N+1)]))
    
def Yule_Simon_distribution(N=20,ru=0.25): #https://en.wikipedia.org/wiki/Yule%E2%80%93Simon_distribution
    def Yule_Simon(k):
        return ru*beta(k,ru+1)
    return list(map(Yule_Simon, [i+1 for i in range(N)]))
    
def Riemann_zeta_function(s=2):#https://en.wikipedia.org/wiki/Riemann_zeta_function
    def Riemann_zeta(k):
        return 1/np.power(k,s)
    
    max = 100
    return np.sum(list(map(Riemann_zeta, [i+1 for i in range(max)])))

def Zeta_distribution(N=16,s=2): #https://en.wikipedia.org/wiki/Zeta_distribution
    def Zeta(k):
        return 1/(Riemann_zeta_function(s=s)*np.power(k,s))
    return list(map(Zeta, [i+1 for i in range(N)]))

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
 
def Chi_distribution(x,k=1): #https://en.wikipedia.org/wiki/Chi_distribution
    return (2/ (np.power(2,k/2)* gamma(k/2)))*np.power(x,k-1)*np.exp(-0.5*x**2)

def Erlang_distribution(x,k=1,u=2.0): #https://en.wikipedia.org/wiki/Erlang_distribution
    return np.power(x,k-1)*np.exp(-1*x/u)/(np.power(u,k)*factorial(k-1))

def Exponential_distribution(x,lam=1): #https://en.wikipedia.org/wiki/Exponential_distribution
    return lam*np.exp(-1*lam*x)

def Von_Mises_distribution(x,u=0,k=0): #https://en.wikipedia.org/wiki/Von_Mises_distribution
    return np.exp(k*np.cos(x-u))/(2*np.pi*Modified_Bessel_functions(alpha=0,x=k))

def Boltzmann_distribution(x,a=1): #https://en.wikipedia.org/wiki/Maxwell%E2%80%93Boltzmann_distribution
    return x**2*np.sqrt(2/np.pi)*np.exp(-0.5*x**2/a**2)/a**3

def Logit_normal_distribution(): #https://en.wikipedia.org/wiki/Logit-normal_distribution
    pass

def Irwin_Hall_distribution(): #https://en.wikipedia.org/wiki/Irwin%E2%80%93Hall_distribution
    pass

def Bates_distribution(): #https://en.wikipedia.org/wiki/Bates_distribution
    pass

def Kumaraswamy_distribution(): #https://en.wikipedia.org/wiki/Kumaraswamy_distribution
    pass

def PERT_distribution(): #https://en.wikipedia.org/wiki/PERT_distribution
    pass

def Reciprocal_distribution(): #https://en.wikipedia.org/wiki/Reciprocal_distribution
    pass

def Triangular_distribution(): #https://en.wikipedia.org/wiki/Triangular_distribution
    pass
def Trapezoidal_distribution(): #https://en.wikipedia.org/wiki/Trapezoidal_distribution
    pass
def Truncated_normal_distribution(): #https://en.wikipedia.org/wiki/Truncated_normal_distribution
    pass
def Wigner_semicircle_distribution(): #https://en.wikipedia.org/wiki/Wigner_semicircle_distribution
    pass
def Dagum_distribution(): #https://en.wikipedia.org/wiki/Dagum_distribution
    pass
def F_distribution(): #https://en.wikipedia.org/wiki/F-distribution
    pass
def Folded_normal_distribution(): #https://en.wikipedia.org/wiki/Folded_normal_distribution
    pass
def Frechet_distribution(): #https://en.wikipedia.org/wiki/Fr%C3%A9chet_distribution
    pass
def Gompertz_distribution(): #https://en.wikipedia.org/wiki/Gompertz_distribution
    pass
def Half_normal_distribution(): #https://en.wikipedia.org/wiki/Half-normal_distribution
    pass
def Nakagami_distribution(): #https://en.wikipedia.org/wiki/Nakagami_distribution
    pass
def Levy_distribution(): #https://en.wikipedia.org/wiki/L%C3%A9vy_distribution
    pass
def Shifted_Gompertz_distribution(): #https://en.wikipedia.org/wiki/Shifted_Gompertz_distribution
    pass
def JohnsonSU_distribution(): #https://en.wikipedia.org/wiki/Johnson%27s_SU-distribution
    pass
def Landau_distribution(): #https://en.wikipedia.org/wiki/Landau_distribution
    pass
def Stable_distribution(): #https://en.wikipedia.org/wiki/Stable_distribution
    pass
def Skew_normal_distribution(): #https://en.wikipedia.org/wiki/Skew_normal_distribution
    pass
def Noncentral_t_distribution(): #https://en.wikipedia.org/wiki/Noncentral_t-distribution
    pass



#------------------------------------main----------------------------------#   
def main():
    pass 
    
if __name__ == '__main__':
    main()

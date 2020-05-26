#Python3 Steven
#common proability distrubution
import numpy as np
import matplotlib.pyplot as plt
from sympy import integrate
import math
from scipy.special import gamma,beta,factorial
from permutationAndCom import permut,combinat

#https://en.wikipedia.org/wiki/Gamma_function
#https://en.wikipedia.org/wiki/Beta_function
#https://en.wikipedia.org/wiki/Factorial

#''''''''''''''''''''''''''''''''''''plot fuc''''''''''''''''''''''''''''''''''''''''''
def plot(x,y):
    plt.plot(x,y)
    plt.show()

def plotSub(x,y,ax=None, aspect=False, label=''):
    ax.plot(x,y,label=label)
    #ax.title.set_text(name)
    if aspect:
        ax.set_aspect(1)
    ax.legend()

def scatterSub(x,y,ax=None,label='',marker='.'):
    ax.scatter(x,y,linewidths=.3, label=label, marker=marker)
    ax.legend()
    
#'''''''''''''''''''''''''''''''''''distribution fuc'''''''''''''''''''''''''''''''''''''
def Uniform_distribution(x,a=1,b=3): #https://en.wikipedia.org/wiki/Uniform_distribution_(continuous)
    return np.zeros(len(x)) + 1/(b-a)

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
 
 
#'''''''''''''''''''''''''''''''''''start plot distribution'''''''''''''''''''''''''''''''''''''
imgSavePath=r'.\images\\'

def testDiscrete_uniform_distribution(i=0):
    N = 5
    x = np.linspace(1.0, 4.0, N)
    ax = plt.subplot(1,1,1)
    plt.title('Discrete_uniform_distribution')
    scatterSub(x, Discrete_uniform_distribution(x,N=N), ax,label='Discrete_uniform_distribution')
    plt.xlim(0, 5)
    plt.ylim(0, 0.3)
    plt.savefig(imgSavePath+'dsitribution{}.png'.format(i)), plt.show() 
    
def testUniform_distribution(i):
    x = np.linspace(1.0, 3.0, 50)
    ax = plt.subplot(1,1,1)
    plt.title('Uniform_distribution')
    plotSub(x, Uniform_distribution(x), ax,label='Uniform_distribution')
    plt.xlim(0, 4)
    plt.ylim(0.35, 0.6)
    plt.savefig(imgSavePath+'dsitribution{}.png'.format(i)), plt.show() 
    
def testBinomial_distribution(i):
    ax = plt.subplot(1,1,1)
    plt.title('Binomial_distribution')
    N = 15
    plotSub(np.arange(N), Binomial_distribution(N=N,p=0.2), ax,label='N=15,p=0.2')
    scatterSub(np.arange(N), Binomial_distribution(N=N,p=0.2), ax,label='N=15,p=0.2')
    
    plotSub(np.arange(N), Binomial_distribution(N=N,p=0.6), ax,label='N=15,p=0.6')
    scatterSub(np.arange(N), Binomial_distribution(N=N,p=0.6), ax,label='N=15,p=0.6')
    N = 20
    plotSub(np.arange(N), Binomial_distribution(N=N,p=0.6), ax,label='N=20,p=0.6')
    scatterSub(np.arange(N), Binomial_distribution(N=N,p=0.6), ax,label='N=20,p=0.6')
    plt.savefig(imgSavePath+'dsitribution{}.png'.format(i))
    plt.show() 
    
def testGeometric_distribution(i):
    ax = plt.subplot(1,1,1)
    plt.title('Geometric_distribution')
    N = 15 
    plotSub(np.arange(N), Geometric_distribution(N=N,p=0.2), ax,label='N=20,p=0.2')
    scatterSub(np.arange(N), Geometric_distribution(N=N,p=0.2), ax,label='N=20,p=0.2')
    
    plotSub(np.arange(N), Geometric_distribution(N=N,p=0.5), ax,label='N=20,p=0.6')
    scatterSub(np.arange(N), Geometric_distribution(N=N,p=0.5), ax,label='N=20,p=0.6')
    
    plotSub(np.arange(N), Geometric_distribution(N=N,p=0.8), ax,label='N=25,p=0.6')
    scatterSub(np.arange(N), Geometric_distribution(N=N,p=0.8), ax,label='N=25,p=0.6')
    plt.savefig(imgSavePath+'dsitribution{}.png'.format(i))
    plt.show()
   
def testHypergeometric_distribution(i):
    ax = plt.subplot(1,1,1)
    plt.title('Hypergeometric_distribution')
    N=25
    K=10
    n=5
    plotSub(np.arange(n), Hypergeometric_distribution(N,K,n), ax,label='N=25,K=10,n=5')
    scatterSub(np.arange(n), Hypergeometric_distribution(N,K,n), ax,label='N=25,K=10,n=5')
    
    plt.savefig(imgSavePath+'dsitribution{}.png'.format(i))
    plt.show()
     
    
def testNormalD(i):
    x = np.linspace(-5.0, 5.0, 100)
    #y = NormalDistribution_pdf(x)
    #plot(x,y)
    
    ax = plt.subplot(1,1,1)
    plt.title('Normal Distribution')
    plotSub(x, NormalDistribution_pdf(x), ax,label='Normal')
    plotSub(x, NormalDistribution_pdf(x,u=-1,delta=0.5), ax,label='u=-1,delta=0.5')
    plotSub(x, NormalDistribution_pdf(x,u=1,delta=2), ax,label='u=1,delta=2')
    plt.savefig(imgSavePath+'dsitribution{}.png'.format(i)),plt.show()
    
def testCauchy(i):
    x = np.linspace(-5.0, 5.0, 100)  
    ax = plt.subplot(1,1,1)
    plt.title('Cauchy Distribution')
    #plotSub(x, Cauchy_pdf(x), ax,label='Cauchy')
    plotSub(x, Cauchy_pdf(x,x0=0,scaler=0.75), ax,label='x0=0,scaler=1')
    plotSub(x, Cauchy_pdf(x,x0=0,scaler=1), ax,label='x0=0,scaler=1')
    #plotSub(x, NormalDistribution_pdf(x), ax,label='Normal')
    plotSub(x, Cauchy_pdf(x,x0=0,scaler=2), ax,label='x0=0,scaler=2')
    plotSub(x, Cauchy_pdf(x,x0=-2,scaler=1), ax,label='x0=-2,scaler=1')
    plt.savefig(imgSavePath+'dsitribution{}.png'.format(i)),plt.show()
  
def testLaplace_distribution(i):
    x = np.linspace(-15.0, 15.0, 100)  
    ax = plt.subplot(1,1,1)
    plt.title('Laplace_distribution')
    plotSub(x, Laplace_distribution(x), ax,label='Laplace_distribution')
    plotSub(x, Laplace_distribution(x,u=0,b=2), ax,label='u=0,b=2')
    plotSub(x, Laplace_distribution(x,u=0,b=4), ax,label='u=0,b=4')
    plotSub(x, Laplace_distribution(x,u=-5,b=4), ax,label='u=5,b=4')
    plt.savefig(imgSavePath+'dsitribution{}.png'.format(i)),plt.show()
 
def testLogistic_distribution(i):
    x = np.linspace(-5.0, 20.0, 100)  
    ax = plt.subplot(1,1,1)
    plt.title('Logistic_distribution')
    plotSub(x, Logistic_distribution(x), ax,label='Logistic_distribution')
    plotSub(x, Logistic_distribution(x,u=5,s=2), ax,label='u=5,b=2')
    plotSub(x, Logistic_distribution(x,u=9,s=3), ax,label='u=9,b=3')
    plotSub(x, Logistic_distribution(x,u=9,s=4), ax,label='u=9,b=4')
    plotSub(x, Logistic_distribution(x,u=6,s=2), ax,label='u=6,b=2')
    plotSub(x, Logistic_distribution(x,u=2,s=1), ax,label='u=2,s=1')
    plt.savefig(imgSavePath+'dsitribution{}.png'.format(i)),plt.show()
    
def testLog_normal_distribution(i):
    x = np.linspace(0, 3.0, 100)  
    ax = plt.subplot(1,1,1)
    plt.title('Log_normal_distribution')
    plotSub(x, Log_normal_distribution(x), ax,label='Log_normal_distribution')
    plotSub(x, Log_normal_distribution(x,delta=0.25), ax,label='delta=0.25')
    plotSub(x, Log_normal_distribution(x,delta=0.5), ax,label='delta=0.5')
    #plotSub(x, Log_normal_distribution(x,delta=1.25), ax,label='delta=1.25')
    plt.savefig(imgSavePath+'dsitribution{}.png'.format(i)),plt.show()

def testWeibull_distribution(i):
    x = np.linspace(0, 2.5, 100)  
    ax = plt.subplot(1,1,1)
    plt.title('Weibull_distribution')
    plotSub(x, Weibull_distribution(x,lamda=1,k=0.5), ax,label='lamda=1,k=0.5')
    plotSub(x, Weibull_distribution(x,lamda=1,k=1), ax,label='lamda=1,k=1')
    plotSub(x, Weibull_distribution(x,lamda=1,k=1.5), ax,label='lamda=1,k=1.5')
    plotSub(x, Weibull_distribution(x,lamda=1,k=5), ax,label='lamda=1,k=5')
    plt.savefig(imgSavePath+'dsitribution{}.png'.format(i)),plt.show()

def testPareto_distribution(i):
    x = np.linspace(0, 5, 100)  
    ax = plt.subplot(1,1,1)
    plt.title('Pareto_distribution')
    plotSub(x, Pareto_distribution(x,alpha=1), ax,label='alpha=1')
    plotSub(x, Pareto_distribution(x,alpha=2), ax,label='alpha=2')
    plotSub(x, Pareto_distribution(x,alpha=3), ax,label='alpha=3')
    plotSub(x, Pareto_distribution(x,alpha=1,Xm=2), ax,label='alpha=1,Xm=2')
    plt.savefig(imgSavePath+'dsitribution{}.png'.format(i)),plt.show()

def testRayleigh_distribution(i):
    x = np.linspace(0, 12, 100)  
    ax = plt.subplot(1,1,1)
    plt.title('Rayleigh_distribution')
    plotSub(x, Rayleigh_distribution(x,delta=1), ax,label='delta=1')
    plotSub(x, Rayleigh_distribution(x,delta=2), ax,label='delta=2')
    plotSub(x, Rayleigh_distribution(x,delta=3), ax,label='delta=3')
    plotSub(x, Rayleigh_distribution(x,delta=4), ax,label='delta=4')
    plt.savefig(imgSavePath+'dsitribution{}.png'.format(i)),plt.show()

def testGamma_distribution(i):
    x = np.linspace(0, 20, 100)  
    ax = plt.subplot(1,1,1)
    plt.title('Gamma_distribution')
    plotSub(x, Gamma_distribution(x,k=2.0,theta=2.0), ax,label='k=2.0,theta=2.0')
    plotSub(x, Gamma_distribution(x,k=3.0,theta=2.0), ax,label='k=3.0,theta=2.0')
    plotSub(x, Gamma_distribution(x,k=5.0,theta=1.0), ax,label='5.0,theta=1.0')
    plotSub(x, Gamma_distribution(x,k=9.0,theta=0.5), ax,label='9.0,theta=0.5')
    plotSub(x, Gamma_distribution(x,k=7.5,theta=1.0), ax,label='k=7.5,theta=1.0')
    plotSub(x, Gamma_distribution(x,k=0.5,theta=1.0), ax,label='k=0.5,theta=1.0')
    plt.savefig(imgSavePath+'dsitribution{}.png'.format(i)),plt.show()

def testStudentT_distribution(i): #float("inf") +00 , float("-inf") -00
    x = np.linspace(-5, 5, 100)  
    ax = plt.subplot(1,1,1)
    plt.title('StudentT_distribution')
    plotSub(x, StudentT_distribution(x), ax,label='v=1')
    plotSub(x, StudentT_distribution(x,v=2), ax,label='v=2')
    plotSub(x, StudentT_distribution(x,v=5), ax,label='v=5')
    plotSub(x, StudentT_distribution(x,v=float('inf')), ax,label='v=+00')
    plt.savefig(imgSavePath+'dsitribution{}.png'.format(i)),plt.show()

def testBeta_distribution(i):
    x = np.linspace(0, 1, 100)  
    ax = plt.subplot(1,1,1)
    plt.title('Beta_distribution')
    plotSub(x, Beta_distribution(x), ax,label='alpha=0.5,ba=0.5')
    plotSub(x, Beta_distribution(x,alpha=5,ba=1), ax,label='alpha=5,ba=1')
    plotSub(x, Beta_distribution(x,alpha=1,ba=3), ax,label='alpha=1,ba=3')
    plotSub(x, Beta_distribution(x,alpha=2,ba=2), ax,label='alpha=2,ba=2')
    plotSub(x, Beta_distribution(x,alpha=2,ba=5), ax,label='alpha=2,ba=5')
    plt.savefig(imgSavePath+'dsitribution{}.png'.format(i)),plt.show()
    
def testGeneralized_logistic_distribution(i):
    x = np.linspace(-5.0, 5.0, 100)  
    ax = plt.subplot(1,1,1)
    plt.title('Generalized_logistic_distribution')
    plotSub(x, Generalized_logistic_distribution(x), ax,label='GLD')
    plotSub(x, Generalized_logistic_distribution(x,alpha=0.5), ax,label='GLD alpha=0.5')
    plotSub(x, Gumbel_distribution(x),ax,label='Gumbel_distribution')
    plt.savefig(imgSavePath+'dsitribution{}.png'.format(i)),plt.show()
    
def main():
    i=0
    testDiscrete_uniform_distribution(i)
    i+=1
    testUniform_distribution(i)
    i+=1
    testBinomial_distribution(i)
    i+=1
    testGeometric_distribution(i)
    i+=1
    testHypergeometric_distribution(i)
    i+=1
    testNormalD(i)
    i+=1
    testCauchy(i)
    i+=1
    testLaplace_distribution(i)
    i+=1
    testGeneralized_logistic_distribution(i)
    i+=1
    testLogistic_distribution(i)
    i+=1
    testLog_normal_distribution(i)
    i+=1
    testWeibull_distribution(i)
    i+=1
    testPareto_distribution(i)
    i+=1
    testRayleigh_distribution(i)
    i+=1
    testGamma_distribution(i)
    i+=1
    testStudentT_distribution(i)
    i+=1
    testBeta_distribution(i)
    i+=1
   
    
if __name__ == '__main__':
    main()

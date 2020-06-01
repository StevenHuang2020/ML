#Python3 Steven
#common proability distrubution
import numpy as np
import matplotlib.pyplot as plt
# from sympy import integrate
# import math
# from scipy.special import gamma,beta,factorial
# from permutationAndCom import permut,combinat
from distributions import *

#''''''''''''''''''''''''''''''''''''plot fuc''''''''''''''''''''''''''''''''''''''''''#
def plot(x,y):
    plt.plot(x,y)
    plt.show()

def plotSub(x,y,ax=None, aspect=False, label='',linestyle='solid',marker=''):
    ax.plot(x,y,label=label,marker=marker,linestyle=linestyle)
    #ax.title.set_text(name)
    if aspect:
        ax.set_aspect(1)
    ax.legend()

def scatterSub(x,y,ax=None,label='',marker='.'):
    ax.scatter(x,y,linewidths=.3, label=label, marker=marker)
    ax.legend()
 
#'''''''''''''''''''''''''''''''''''start plot distribution''''''''''''''''''''#
#---------------------------Discrete distribution------------------------------#
imgSavePath=r'.\images\\'

def testDiscrete_uniform_distribution(i=0):
    N = 5
    x = np.arange(1,N)
    ax = plt.subplot(1,1,1)
    plt.title('Discrete_uniform_distribution')
    scatterSub(x, Discrete_uniform_distribution(x,N=N), ax,label='Discrete_uniform_distribution')
    plt.xlim(0, 5)
    plt.ylim(0, 0.3)
    plt.savefig(imgSavePath+'dsitribution{}.png'.format(i)), plt.show() 
    
def testBinomial_distribution(i):
    ax = plt.subplot(1,1,1)
    plt.title('Binomial_distribution')
    N = 15
    x = np.arange(N)
    plotSub(x, Binomial_distribution(N=N,p=0.2), ax,label='N=15,p=0.2')
    scatterSub(x, Binomial_distribution(N=N,p=0.2), ax,label='N=15,p=0.2')
    
    plotSub(x, Binomial_distribution(N=N,p=0.6), ax,label='N=15,p=0.6')
    scatterSub(x, Binomial_distribution(N=N,p=0.6), ax,label='N=15,p=0.6')
    N = 20
    x = np.arange(N)
    plotSub(x, Binomial_distribution(N=N,p=0.6), ax,label='N=20,p=0.6')
    scatterSub(x, Binomial_distribution(N=N,p=0.6), ax,label='N=20,p=0.6')
    plt.savefig(imgSavePath+'dsitribution{}.png'.format(i)), plt.show() 
    
def testPoisson_distribution(i):
    ax = plt.subplot(1,1,1)
    plt.title('Poisson_distribution')
    N=20
    x = np.arange(N)
    plotSub(x, Poisson_distribution(N=N), ax,label='N=20,lam=1')
    scatterSub(x, Poisson_distribution(N=N), ax,label='N=20,lam=1')
    
    plotSub(x, Poisson_distribution(N=N,lam=2), ax,label='N=20,lam=2')
    scatterSub(x, Poisson_distribution(N=N,lam=2), ax,label='N=20,lam=2')
    plotSub(x, Poisson_distribution(N=N,lam=4), ax,label='N=20,lam=4')
    scatterSub(x, Poisson_distribution(N=N,lam=4), ax,label='N=20,lam=4')
    plt.savefig(imgSavePath+'dsitribution{}.png'.format(i)), plt.show()
    
def testGeometric_distribution(i):
    ax = plt.subplot(1,1,1)
    plt.title('Geometric_distribution')
    N = 15 
    x = np.arange(N)
    plotSub(x, Geometric_distribution(N=N,p=0.2), ax,label='N=20,p=0.2')
    scatterSub(x, Geometric_distribution(N=N,p=0.2), ax,label='N=20,p=0.2')
    
    plotSub(x, Geometric_distribution(N=N,p=0.5), ax,label='N=20,p=0.6')
    scatterSub(x, Geometric_distribution(N=N,p=0.5), ax,label='N=20,p=0.6')
    
    plotSub(x, Geometric_distribution(N=N,p=0.8), ax,label='N=25,p=0.6')
    scatterSub(x, Geometric_distribution(N=N,p=0.8), ax,label='N=25,p=0.6')
    plt.savefig(imgSavePath+'dsitribution{}.png'.format(i)), plt.show()
   
def testHypergeometric_distribution(i):
    ax = plt.subplot(1,1,1)
    plt.title('Hypergeometric_distribution')

    n = 100
    x = np.arange(n)
    plotSub(x, Hypergeometric_distribution(N=500,K=50,n=n), ax,label='N=500,K=50,n=100')
    scatterSub(x, Hypergeometric_distribution(N=500,K=50,n=n), ax,label='N=500,K=50,n=100')
    
    n = 200
    x = np.arange(n)
    plotSub(x, Hypergeometric_distribution(N=500,K=60,n=n), ax,label='N=500,K=60,n=200')
    scatterSub(x, Hypergeometric_distribution(N=500,K=60,n=n), ax,label='N=500,K=60,n=200')
    
    n = 300
    x = np.arange(n)
    plotSub(x, Hypergeometric_distribution(N=500,K=70,n=n), ax,label='N=500,K=70,n=300')
    scatterSub(x, Hypergeometric_distribution(N=500,K=70,n=n), ax,label='N=500,K=70,n=300')
    plt.xlim(0, 65)
    plt.savefig(imgSavePath+'dsitribution{}.png'.format(i)), plt.show()

def testZipfsLaw(i):
    ax = plt.subplot(1,1,1)
    plt.title('ZipfsLaw')
    N=10
    x = np.arange(N)
    plotSub(x, ZipfsLaw(), ax,label='N=10, s=1')
    scatterSub(x, ZipfsLaw(), ax,label='N=10, s=1')
    
    plotSub(x, ZipfsLaw(N=10, s=2), ax,label='N=10, s=2')
    scatterSub(x, ZipfsLaw(N=10, s=2), ax,label='N=10, s=2')
    
    plotSub(x, ZipfsLaw(N=10, s=3), ax,label='N=10, s=3')
    scatterSub(x, ZipfsLaw(N=10, s=3), ax,label='N=10, s=3')
    
    plotSub(x, ZipfsLaw(N=10, s=4), ax,label='N=10, s=4')
    scatterSub(x, ZipfsLaw(N=10, s=4), ax,label='N=10, s=4')
    #N=20
    #plotSub(x, ZipfsLaw(N=20, s=2), ax,label='N=20, s=2')
    plt.yscale("log")
    plt.savefig(imgSavePath+'dsitribution{}.png'.format(i))
    plt.show() 
        
def testBeta_binomial_distribution(i):
    ax = plt.subplot(1,1,1)
    plt.title('Beta_binomial_distribution')
    N=20
    x = np.arange(N)
    plotSub(x, Beta_binomial_distribution(N=N), ax,label='alpha=0.2,bta=0.25')
    scatterSub(x, Beta_binomial_distribution(N=N), ax,label='alpha=0.2,bta=0.25')
    
    plotSub(x, Beta_binomial_distribution(N=N,alpha=0.7,bta=2), ax,label='alpha=0.7,bta=2')
    scatterSub(x, Beta_binomial_distribution(N=N,alpha=0.7,bta=2), ax,label='alpha=0.7,bta=2')
    
    plotSub(x, Beta_binomial_distribution(N=N,alpha=2,bta=2), ax,label='alpha=2,bta=2')
    scatterSub(x, Beta_binomial_distribution(N=N,alpha=2,bta=2), ax,label='alpha=2,bta=2')
    
    plotSub(x, Beta_binomial_distribution(N=N,alpha=600,bta=400), ax,label='alpha=600,bta=400')
    scatterSub(x, Beta_binomial_distribution(N=N,alpha=600,bta=400), ax,label='alpha=600,bta=400')
    plt.savefig(imgSavePath+'dsitribution{}.png'.format(i)), plt.show()
     
def testLogarithmic_distribution(i):
    ax = plt.subplot(1,1,1)
    plt.title('Logarithmic_distribution')
    N=10
    x = np.arange(N)
    plotSub(x, Logarithmic_distribution(N=N), ax,label='p=0.33')
    scatterSub(x, Logarithmic_distribution(N=N), ax,label='p=0.33')
    plotSub(x, Logarithmic_distribution(N=N,p=0.66), ax,label='p=0.66')
    scatterSub(x, Logarithmic_distribution(N=N,p=0.66), ax,label='p=0.66')
    plotSub(x, Logarithmic_distribution(N=N,p=0.99), ax,label='p=0.99')
    scatterSub(x, Logarithmic_distribution(N=N,p=0.99), ax,label='p=0.99')    
    plt.savefig(imgSavePath+'dsitribution{}.png'.format(i)), plt.show()
     
def testConway_Maxwell_Poisson_distribution(i):
    ax = plt.subplot(1,1,1)
    plt.title('Conway_Maxwell_Poisson_distribution')
    N=20
    x = np.arange(N)
    plotSub(x, Conway_Maxwell_Poisson_distribution(N=N), ax,label='lam=1, v=1.5')
    scatterSub(x, Conway_Maxwell_Poisson_distribution(N=N), ax,label='lam=1, v=1.5')
    plotSub(x, Conway_Maxwell_Poisson_distribution(N=N,lam=3, v=1.1), ax,label='lam=3, v=1.1')
    scatterSub(x, Conway_Maxwell_Poisson_distribution(N=N,lam=3, v=1.1), ax,label='lam=3, v=1.1')
    plotSub(x, Conway_Maxwell_Poisson_distribution(N=N,lam=5, v=0.7), ax,label='lam=5, v=0.7')
    scatterSub(x, Conway_Maxwell_Poisson_distribution(N=N,lam=5, v=0.7), ax,label='lam=5, v=0.7')    
    plt.savefig(imgSavePath+'dsitribution{}.png'.format(i)), plt.show()
     
def testSkellam_distribution(i):
    ax = plt.subplot(1,1,1)
    plt.title('Skellam_distribution')
    N=16
    x = np.arange(N+1)-N/2
    plotSub(x, Skellam_distribution(N=N), ax,label='u1=1,u2=1')
    scatterSub(x, Skellam_distribution(N=N), ax,label='u1=1,u2=1')
    plotSub(x, Skellam_distribution(N=N,u1=2,u2=2), ax,label='u1=2,u2=2')
    scatterSub(x, Skellam_distribution(N=N,u1=2,u2=2), ax,label='u1=2,u2=2')
    plotSub(x, Skellam_distribution(N=N,u1=3,u2=3), ax,label='u1=3,u2=3')
    scatterSub(x, Skellam_distribution(N=N,u1=3,u2=3), ax,label='u1=3,u2=3')
    plotSub(x, Skellam_distribution(N=N,u1=1,u2=3), ax,label='u1=1,u2=3')
    scatterSub(x, Skellam_distribution(N=N,u1=1,u2=3), ax,label='u1=1,u2=3') 
    plt.savefig(imgSavePath+'dsitribution{}.png'.format(i)), plt.show()

def testYule_Simon_distribution(i):
    ax = plt.subplot(1,1,1)
    plt.title('Yule_Simon_distribution')
    N=20
    x = np.arange(N)+1
    plotSub(x, Yule_Simon_distribution(N=N), ax,label='ru=0.25')
    scatterSub(x, Yule_Simon_distribution(N=N), ax,label='ru=0.25')
    plotSub(x, Yule_Simon_distribution(N=N,ru=0.5), ax,label='ru=0.5')
    scatterSub(x, Yule_Simon_distribution(N=N,ru=0.5), ax,label='ru=0.5')
    plotSub(x, Yule_Simon_distribution(N=N,ru=1), ax,label='ru=1')
    scatterSub(x, Yule_Simon_distribution(N=N,ru=1), ax,label='ru=1') 
    plotSub(x, Yule_Simon_distribution(N=N,ru=2), ax,label='ru=2')
    scatterSub(x, Yule_Simon_distribution(N=N,ru=2), ax,label='ru=2') 
    plotSub(x, Yule_Simon_distribution(N=N,ru=4), ax,label='ru=4')
    scatterSub(x, Yule_Simon_distribution(N=N,ru=4), ax,label='ru=4') 

    plt.yscale("log")
    plt.savefig(imgSavePath+'dsitribution{}.png'.format(i))
    plt.show()
     
def testZeta_distribution(i):
    ax = plt.subplot(1,1,1)
    plt.title('Zeta_distribution')
    N=16
    x = np.arange(N)+1
    plotSub(x, Zeta_distribution(N=N), ax,label='s=2')
    scatterSub(x, Zeta_distribution(N=N), ax,label='s=2')
    plotSub(x, Zeta_distribution(N=N,s=3), ax,label='s=3')
    scatterSub(x, Zeta_distribution(N=N,s=3), ax,label='s=3')
    plotSub(x, Zeta_distribution(N=N,s=4), ax,label='s=4')
    scatterSub(x, Zeta_distribution(N=N,s=4), ax,label='s=4')
    plotSub(x, Zeta_distribution(N=N,s=5), ax,label='s=5')
    scatterSub(x, Zeta_distribution(N=N,s=5), ax,label='s=5') 
    plt.yscale("log")
    plt.savefig(imgSavePath+'dsitribution{}.png'.format(i))
    plt.show()
    
#---------------------------Continuous distribution------------------------------#     
def testUniform_distribution(i):
    x = np.linspace(1.0, 3.0, 50)
    ax = plt.subplot(1,1,1)
    plt.title('Uniform_distribution')
    plotSub(x, Uniform_distribution(x), ax,label='Uniform_distribution')
    plt.xlim(0, 4)
    plt.ylim(0.35, 0.6)
    plt.savefig(imgSavePath+'dsitribution{}.png'.format(i)), plt.show() 
    
def testNormalDistribution(i):
    x = np.linspace(-5.0, 5.0, 100)
    #y = NormalDistribution(x)
    #plot(x,y)
    
    ax = plt.subplot(1,1,1)
    plt.title('Normal Distribution')
    plotSub(x, NormalDistribution(x), ax,label='u=0,delta=1')
    plotSub(x, NormalDistribution(x,u=-1,delta=0.5), ax,label='u=-1,delta=0.5')
    plotSub(x, NormalDistribution(x,u=1,delta=2), ax,label='u=1,delta=2')
    plt.savefig(imgSavePath+'dsitribution{}.png'.format(i)),plt.show()
    
def testCauchy(i):
    x = np.linspace(-5.0, 5.0, 100)  
    ax = plt.subplot(1,1,1)
    plt.title('Cauchy Distribution')
    #plotSub(x, Cauchy_pdf(x), ax,label='Cauchy')
    plotSub(x, Cauchy_pdf(x,x0=0,scaler=0.75), ax,label='x0=0,scaler=1')
    plotSub(x, Cauchy_pdf(x,x0=0,scaler=1), ax,label='x0=0,scaler=1')
    #plotSub(x, NormalDistribution(x), ax,label='Normal')
    plotSub(x, Cauchy_pdf(x,x0=0,scaler=2), ax,label='x0=0,scaler=2')
    plotSub(x, Cauchy_pdf(x,x0=-2,scaler=1), ax,label='x0=-2,scaler=1')
    plt.savefig(imgSavePath+'dsitribution{}.png'.format(i)),plt.show()
  
def testLaplace_distribution(i):
    x = np.linspace(-15.0, 15.0, 100)  
    ax = plt.subplot(1,1,1)
    plt.title('Laplace_distribution')
    plotSub(x, Laplace_distribution(x), ax,label='u=0,b=1')
    plotSub(x, Laplace_distribution(x,u=0,b=2), ax,label='u=0,b=2')
    plotSub(x, Laplace_distribution(x,u=0,b=4), ax,label='u=0,b=4')
    plotSub(x, Laplace_distribution(x,u=-5,b=4), ax,label='u=5,b=4')
    plt.savefig(imgSavePath+'dsitribution{}.png'.format(i)),plt.show()
 
def testLogistic_distribution(i):
    x = np.linspace(-5.0, 20.0, 100)  
    ax = plt.subplot(1,1,1)
    plt.title('Logistic_distribution')
    plotSub(x, Logistic_distribution(x), ax,label='u=5,s=1')
    plotSub(x, Logistic_distribution(x,u=5,s=2), ax,label='u=5,s=2')
    plotSub(x, Logistic_distribution(x,u=9,s=3), ax,label='u=9,s=3')
    plotSub(x, Logistic_distribution(x,u=9,s=4), ax,label='u=9,s=4')
    plotSub(x, Logistic_distribution(x,u=6,s=2), ax,label='u=6,s=2')
    plotSub(x, Logistic_distribution(x,u=2,s=1), ax,label='u=2,s=1')
    plt.savefig(imgSavePath+'dsitribution{}.png'.format(i)),plt.show()
    
def testLog_normal_distribution(i):
    x = np.linspace(0, 3.0, 100)  
    ax = plt.subplot(1,1,1)
    plt.title('Log_normal_distribution')
    plotSub(x, Log_normal_distribution(x), ax,label='delta=1.0')
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
    plotSub(x, Beta_distribution(x), ax,label='alpha=0.5,bta=0.5')
    plotSub(x, Beta_distribution(x,alpha=5,bta=1), ax,label='alpha=5,bta=1')
    plotSub(x, Beta_distribution(x,alpha=1,bta=3), ax,label='alpha=1,bta=3')
    plotSub(x, Beta_distribution(x,alpha=2,bta=2), ax,label='alpha=2,bta=2')
    plotSub(x, Beta_distribution(x,alpha=2,bta=5), ax,label='alpha=2,bta=5')
    plt.savefig(imgSavePath+'dsitribution{}.png'.format(i)),plt.show()
    
def testGeneralized_logistic_distribution(i):
    x = np.linspace(-5.0, 5.0, 100)  
    ax = plt.subplot(1,1,1)
    plt.title('Generalized_logistic_distribution')
    plotSub(x, Generalized_logistic_distribution(x), ax,label='alpha=1.0')
    plotSub(x, Generalized_logistic_distribution(x,alpha=0.5), ax,label='alpha=0.5')
    plt.savefig(imgSavePath+'dsitribution{}.png'.format(i)),plt.show()

def testGumbel_distribution(i):
    x = np.linspace(-5.0, 20.0, 100)  
    ax = plt.subplot(1,1,1)
    plt.title('Gumbel_distribution')
    plotSub(x, Gumbel_distribution(x), ax,label='u=0.5,belta=2.0')
    plotSub(x, Gumbel_distribution(x,u=1.0,belta=2.0), ax,label='u=0.5,belta=2.0')
    plotSub(x, Gumbel_distribution(x,u=1.5,belta=3.0),ax,label='u=1.5,belta=3.0')
    plotSub(x, Gumbel_distribution(x,u=3.0,belta=4.0),ax,label='u=3.0,belta=4.0')
    plt.savefig(imgSavePath+'dsitribution{}.png'.format(i)),plt.show()    

def testChi_distribution(i):
    x = np.linspace(0, 4.0, 100)  
    ax = plt.subplot(1,1,1)
    plt.title('Chi_distribution')
    plotSub(x, Chi_distribution(x), ax,label='k=1')
    plotSub(x, Chi_distribution(x,k=2), ax,label='k=2')
    plotSub(x, Chi_distribution(x,k=3),ax,label='k=3')
    plotSub(x, Chi_distribution(x,k=4),ax,label='k=4')
    plt.savefig(imgSavePath+'dsitribution{}.png'.format(i)),plt.show()   
    
def testErlang_distribution(i):
    x = np.linspace(0, 20.0, 100)  
    ax = plt.subplot(1,1,1)
    plt.title('Erlang_distribution')
    plotSub(x, Erlang_distribution(x,k=1,u=2.0), ax,label='k=1,u=2.0')
    plotSub(x, Erlang_distribution(x,k=2,u=2.0),ax,label='k=2,u=2.0')
    plotSub(x, Erlang_distribution(x,k=3,u=2.0),ax,label='k=3,u=2.0')
    plotSub(x, Erlang_distribution(x,k=5,u=1.0), ax,label='k=5,u=1.0')
    plotSub(x, Erlang_distribution(x,k=7,u=0.5),ax,label='k=7,u=0.5')
    plotSub(x, Erlang_distribution(x,k=9,u=1.0),ax,label='k=9,u=1.0')
    plotSub(x, Erlang_distribution(x,k=1,u=1.0),ax,label='k=1,u=1.0')
    plt.savefig(imgSavePath+'dsitribution{}.png'.format(i)),plt.show()   
    
def testExponential_distribution(i):
    x = np.linspace(0, 4.0, 100)  
    ax = plt.subplot(1,1,1)
    plt.title('Exponential_distribution')
    plotSub(x, Exponential_distribution(x), ax,label='lam=1')
    plotSub(x, Exponential_distribution(x,lam=0.5), ax,label='lam=0.5')
    plotSub(x, Exponential_distribution(x,lam=1.5), ax,label='lam=1.5')
    plt.savefig(imgSavePath+'dsitribution{}.png'.format(i)),plt.show()   
    
def testBoltzmann_distribution(i):
    x = np.linspace(0, 20.0, 100)  
    ax = plt.subplot(1,1,1)
    plt.title('Boltzmann_distribution')
    plotSub(x, Boltzmann_distribution(x), ax,label='a=1')
    plotSub(x, Boltzmann_distribution(x,a=2), ax,label='a=2')
    plotSub(x, Boltzmann_distribution(x,a=5), ax,label='a=5')
    plt.savefig(imgSavePath+'dsitribution{}.png'.format(i)),plt.show()   

def testVon_Mises_distribution(i):
    x = np.linspace(-np.pi, np.pi, 100)  
    ax = plt.subplot(1,1,1)
    plt.title('Von_Mises_distribution')
    plotSub(x, Von_Mises_distribution(x), ax,label='u=0,k=0')
    plotSub(x, Von_Mises_distribution(x,u=0,k=0.5), ax,label='u=0,k=0.5')
    plotSub(x, Von_Mises_distribution(x,u=0,k=1), ax,label='u=0,k=1')
    plotSub(x, Von_Mises_distribution(x,u=0,k=2), ax,label='u=0,k=2')
    plotSub(x, Von_Mises_distribution(x,u=0,k=4), ax,label='u=0,k=4')
    #plotSub(x, Von_Mises_distribution(x,u=1,k=8), ax,label='u=1,k=8')
    plt.savefig(imgSavePath+'dsitribution{}.png'.format(i))
    ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi ))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 2))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(plt.FormatStrFormatter('%d $\pi$')))
    plt.show()   
    
def testLogit_normal_distribution(i):
    x = np.linspace(0, 1.0, 100)  
    ax = plt.subplot(1,1,1)
    plt.title('Logit_normal_distribution')
    plotSub(x, Logit_normal_distribution(x), ax,label='deta=0.2,u=0')
    plotSub(x, Logit_normal_distribution(x,deta=0.3,u=0), ax,label='deta=0.3,u=0')
    plotSub(x, Logit_normal_distribution(x,deta=0.5,u=0), ax,label='deta=0.5,u=0')
    plotSub(x, Logit_normal_distribution(x,deta=1.0,u=0), ax,label='deta=1.0,u=0')
    plotSub(x, Logit_normal_distribution(x,deta=1.5,u=0), ax,label='deta=1.5,u=0')
    plotSub(x, Logit_normal_distribution(x,deta=3.0,u=0), ax,label='deta=3.0,u=0')
    plt.savefig(imgSavePath+'dsitribution{}.png'.format(i)),plt.show()
    
def testIrwin_Hall_distribution(i):
    total=100
    ax = plt.subplot(1,1,1)
    plt.title('Irwin_Hall_distribution')
    n=1
    x = np.linspace(0, n, total)
    plotSub(x, Irwin_Hall_distribution(x), ax,label='n=1')
    n=2
    x = np.linspace(0, n, total)
    
    #print('x=',x)
    #print('res=',Irwin_Hall_distribution(x,n=2))
    plotSub(x, Irwin_Hall_distribution(x,n=2), ax,label='n=2')
    n=4
    x = np.linspace(0, n, total)
    plotSub(x, Irwin_Hall_distribution(x,n=4), ax,label='n=4')
    n=8
    x = np.linspace(0, n, total)
    plotSub(x, Irwin_Hall_distribution(x,n=8), ax,label='n=8')
    n=16
    x = np.linspace(0, n, total)
    plotSub(x, Irwin_Hall_distribution(x,n=16), ax,label='n=16')
    plt.savefig(imgSavePath+'dsitribution{}.png'.format(i)),plt.show()

def testBates_distribution(i):
    ax = plt.subplot(1,1,1)
    plt.title('Bates_distribution')
    x = np.linspace(0, 1, 100)
    
    plotSub(x, Bates_distribution(x), ax,label='n=1')
    plotSub(x, Bates_distribution(x,n=2), ax,label='n=2')
    plotSub(x, Bates_distribution(x,n=3), ax,label='n=3')
    plotSub(x, Bates_distribution(x,n=10), ax,label='n=10')
    plotSub(x, Bates_distribution(x,n=20), ax,label='n=20')
    plt.savefig(imgSavePath+'dsitribution{}.png'.format(i)),plt.show()
    
def testKumaraswamy_distribution(i):
    ax = plt.subplot(1,1,1)
    plt.title('Kumaraswamy_distribution')
    x = np.linspace(0, 1, 100)
    
    plotSub(x, Kumaraswamy_distribution(x), ax,label='a=0.5,b=0.5')
    plotSub(x, Kumaraswamy_distribution(x,a=5.0,b=1.0), ax,label='a=5.0,b=1.0')
    plotSub(x, Kumaraswamy_distribution(x,a=1.0,b=3.0), ax,label='a=1.0,b=3.0')
    plotSub(x, Kumaraswamy_distribution(x,a=2.0,b=2.0), ax,label='a=2.0,b=2.0')
    plotSub(x, Kumaraswamy_distribution(x,a=2.0,b=5.0), ax,label='a=2.0,b=5.0')
    plt.savefig(imgSavePath+'dsitribution{}.png'.format(i)),plt.show()
    
def testPERT_distribution(i):
    ax = plt.subplot(1,1,1)
    plt.title('PERT_distribution')
    
    a=0
    b=10
    c=100
    x = np.linspace(a, c, 100)
    
    plotSub(x, PERT_distribution(x), ax,label='a=0,b=10,c=100')
    plotSub(x, PERT_distribution(x,a=0,b=50,c=100), ax,label='a=0,b=50,c=100')
    plotSub(x, PERT_distribution(x,a=0,b=70,c=100), ax,label='a=0,b=70,c=100')
    plotSub(x, PERT_distribution(x,a=0,b=90,c=100), ax,label='a=0,b=90,c=100')
    plt.savefig(imgSavePath+'dsitribution{}.png'.format(i)),plt.show()
    
def testReciprocal_distribution(i):
    ax = plt.subplot(1,1,1)
    plt.title('Reciprocal_distribution')
    a=1
    b=4
    x = np.linspace(a, b, 100)    
    plotSub(x, Reciprocal_distribution(x,a=a,b=b), ax,label='a=1,b=4')
    a=2
    b=6
    x = np.linspace(a, b, 100)
    plotSub(x, Reciprocal_distribution(x,a=a,b=b), ax,label='a=2,b=6')
    a=1
    b=5
    x = np.linspace(a, b, 100)
    plotSub(x, Reciprocal_distribution(x,a=a,b=b), ax,label='a=1,b=5')
    plt.savefig(imgSavePath+'dsitribution{}.png'.format(i)),plt.show()
    
def testTriangular_distribution(i):
    ax = plt.subplot(1,1,1)
    plt.title('Triangular_distribution')
    x = np.linspace(0, 5, 100)
    plotSub(x, Triangular_distribution(x), ax,label='a=1,c=3,b=4')
    plt.savefig(imgSavePath+'dsitribution{}.png'.format(i)),plt.show()
    
def testTrapezoidal_distribution(i):
    ax = plt.subplot(1,1,1)
    plt.title('Trapezoidal_distribution')
    a=-3
    d=0
    x = np.linspace(a, d, 100)
    plotSub(x, Trapezoidal_distribution(x,a=a,b=-2,c=-1,d=d), ax,label='a=-3,b=-2,c=-1,d=0')
    a=1
    d=5
    x = np.linspace(a, d, 100)
    plotSub(x, Trapezoidal_distribution(x,a=a,b=2,c=4.5,d=d), ax,label='a=1,b=2,c=4.5,d=5')
    a=-2
    d=2
    x = np.linspace(a, d, 100)
    plotSub(x, Trapezoidal_distribution(x,a=-2,b=0,c=1,d=2), ax,label='a=-2,b=0,c=1,d=2')
    
    plt.savefig(imgSavePath+'dsitribution{}.png'.format(i)),plt.show()
    
def testWigner_semicircle_distribution(i):
    ax = plt.subplot(1,1,1)
    plt.title('Wigner_semicircle_distribution')
    r = 0.25
    x = np.linspace(-r, r, 100)
    plotSub(x, Wigner_semicircle_distribution(x,r=r), ax,label='r=0.25')
    r = 0.5
    x = np.linspace(-r, r, 100)
    plotSub(x, Wigner_semicircle_distribution(x,r=r), ax,label='r=0.5')
    r = 1.0
    x = np.linspace(-r, r, 100)
    plotSub(x, Wigner_semicircle_distribution(x,r=r), ax,label='r=1.0')
    r = 2.0
    x = np.linspace(-r, r, 100)
    plotSub(x, Wigner_semicircle_distribution(x,r=r), ax,label='r=2.0')
    r = 3.0
    x = np.linspace(-r, r, 100)
    plotSub(x, Wigner_semicircle_distribution(x,r=r), ax,label='r=3.0')
    plt.savefig(imgSavePath+'dsitribution{}.png'.format(i)),plt.show()
    
def testF_distribution(i):
    ax = plt.subplot(1,1,1)
    plt.title('F_distribution')
    x = np.linspace(0, 5, 100)
    plotSub(x, F_distribution(x), ax,label='d1=1,d2=1')
    plotSub(x, F_distribution(x,d1=2,d2=1), ax,label='d1=2,d2=1')
    plotSub(x, F_distribution(x,d1=5,d2=2), ax,label='d1=5,d2=1')
    plotSub(x, F_distribution(x,d1=10,d2=1), ax,label='d1=10,d2=1')
    plotSub(x, F_distribution(x,d1=20,d2=20), ax,label='d1=20,d2=20')
    plt.savefig(imgSavePath+'dsitribution{}.png'.format(i)),plt.show()
    
def testLandau_distribution(i):
    ax = plt.subplot(1,1,1)
    plt.title('Landau_distribution')
    y=[]
    x = np.linspace(-5, 15, 200)
    for _ in x:
        y.append(Landau_distribution(_,u=0,c=1))
    
    #print('y=',y)
    plotSub(x, y, ax,label='u=0,c=1')
    plt.savefig(imgSavePath+'dsitribution{}.png'.format(i)),plt.show()
    
def testJohnsonSU_distribution(i):
    ax = plt.subplot(1,1,1)
    plt.title('JohnsonSU_distribution')
    x = np.linspace(-6, 12, 100)
    plotSub(x, JohnsonSU_distribution(x), ax,label='gama=-2,deta=2,xi=1.1,lam=1.5')
    plotSub(x, JohnsonSU_distribution(x,gama=1,deta=2,xi=1.1,lam=1.5), ax,label='gama=1,deta=2,xi=1.1,lam=1.5')
    plotSub(x, JohnsonSU_distribution(x,gama=3,deta=2,xi=1.1,lam=1.5), ax,label='gama=3,deta=2,xi=1.1,lam=1.5')
    plt.savefig(imgSavePath+'dsitribution{}.png'.format(i)),plt.show()
    
def testShifted_Gompertz_distribution(i):
    ax = plt.subplot(1,1,1)
    plt.title('Shifted_Gompertz_distribution')
    x = np.linspace(0, 15, 100)
    plotSub(x, Shifted_Gompertz_distribution(x), ax,label='b=0.4,ni=0.01')
    plotSub(x, Shifted_Gompertz_distribution(x,b=0.4,ni=0.10), ax,label='b=0.4,ni=0.10')
    plotSub(x, Shifted_Gompertz_distribution(x,b=0.4,ni=0.50), ax,label='b=0.4,ni=0.50')
    plotSub(x, Shifted_Gompertz_distribution(x,b=0.4,ni=1.00), ax,label='b=0.4,ni=1.00')
    plotSub(x, Shifted_Gompertz_distribution(x,b=0.4,ni=10.0), ax,label='b=0.4,ni=10.0')
    plotSub(x, Shifted_Gompertz_distribution(x,b=0.8,ni=10.0), ax,label='b=0.8,ni=10.0')
    plt.savefig(imgSavePath+'dsitribution{}.png'.format(i)),plt.show()
    
def testLevy_distribution(i):
    ax = plt.subplot(1,1,1)
    plt.title('Levy_distribution')
    x = np.linspace(0, 3, 100)
    plotSub(x, Levy_distribution(x), ax,label='u=0,c=0.5')
    plotSub(x, Levy_distribution(x,u=0,c=1), ax,label='u=0,c=1')
    plotSub(x, Levy_distribution(x,u=1,c=1), ax,label='u=1,c=1')
    plotSub(x, Levy_distribution(x,u=0,c=2), ax,label='u=0,c=2')
    plotSub(x, Levy_distribution(x,u=0,c=4), ax,label='u=0,c=4')
    plotSub(x, Levy_distribution(x,u=0,c=8), ax,label='u=0,c=8')
    plt.savefig(imgSavePath+'dsitribution{}.png'.format(i)),plt.show()
    
def testNakagami_distribution(i):
    ax = plt.subplot(1,1,1)
    plt.title('Nakagami_distribution')
    x = np.linspace(0, 3, 100)
    plotSub(x, Nakagami_distribution(x), ax,label='m=0.5,w=1')
    plotSub(x, Nakagami_distribution(x,m=1,w=1), ax,label='m=1,w=1')
    plotSub(x, Nakagami_distribution(x,m=1,w=2), ax,label='m=1,w=2')
    plotSub(x, Nakagami_distribution(x,m=1,w=3), ax,label='m=1,w=3')
    plotSub(x, Nakagami_distribution(x,m=2,w=1), ax,label='m=2,w=1')
    plotSub(x, Nakagami_distribution(x,m=2,w=2), ax,label='m=2,w=2')
    plotSub(x, Nakagami_distribution(x,m=5,w=1), ax,label='m=5,w=1')
    plt.savefig(imgSavePath+'dsitribution{}.png'.format(i)),plt.show()
    
def testGompertz_distribution(i):
    ax = plt.subplot(1,1,1)
    plt.title('Gompertz_distribution')
    x = np.linspace(0, 5, 100)
    plotSub(x, Gompertz_distribution(x), ax,label='eta=0.1,b=1')
    plotSub(x, Gompertz_distribution(x,eta=2.0,b=1), ax,label='eta=2.0,b=1')
    plotSub(x, Gompertz_distribution(x,eta=3.0,b=1), ax,label='eta=3.0,b=1')
    plotSub(x, Gompertz_distribution(x,eta=1.0,b=2), ax,label='eta=1.0,b=2')
    plotSub(x, Gompertz_distribution(x,eta=1.0,b=3), ax,label='eta=1.0,b=3')
    plt.savefig(imgSavePath+'dsitribution{}.png'.format(i)),plt.show()
    
def testFrechet_distribution(i):
    ax = plt.subplot(1,1,1)
    plt.title('Frechet_distribution')
    x = np.linspace(0, 4, 100)
    plotSub(x, Frechet_distribution(x), ax,label='alpha=1,s=1,m=0')
    plotSub(x, Frechet_distribution(x,alpha=1,s=1,m=0), ax,label='alpha=1,s=1,m=0')
    plotSub(x, Frechet_distribution(x,alpha=2,s=1,m=0), ax,label='alpha=2,s=1,m=0')
    plotSub(x, Frechet_distribution(x,alpha=3,s=1,m=0), ax,label='alpha=3,s=1,m=0')
    plotSub(x, Frechet_distribution(x,alpha=1,s=2,m=0), ax,label='alpha=1,s=2,m=0')
    plotSub(x, Frechet_distribution(x,alpha=2,s=2,m=0), ax,label='alpha=2,s=2,m=0')
    plotSub(x, Frechet_distribution(x,alpha=3,s=2,m=0), ax,label='alpha=3,s=2,m=0')
    plt.savefig(imgSavePath+'dsitribution{}.png'.format(i)),plt.show()
    
def testFolded_normal_distribution(i):
    ax = plt.subplot(1,1,1)
    plt.title('Folded_normal_distribution')
    x = np.linspace(0, 5, 100)
    plotSub(x, Folded_normal_distribution(x), ax,label='Folded normal distribution')
    x = np.linspace(-3, 5, 100)
    plotSub(x, NormalDistribution(x,delta=1,u=1), ax,label='Normal distribution')
    
    y = np.linspace(0, 0.55, 30)
    x = np.zeros((len(y),))
    plotSub(x, y, ax,label='',linestyle='dotted')
    plt.savefig(imgSavePath+'dsitribution{}.png'.format(i)),plt.show()

def testHalf_normal_distribution(i):
    ax = plt.subplot(1,1,1)
    plt.title('Half_normal_distribution')
    x = np.linspace(0, 5, 100)
    plotSub(x, Half_normal_distribution(x), ax,label='Half normal distribution')
    plotSub(x, Folded_normal_distribution(x), ax,label='Folded normal distribution')
    
    x = np.linspace(-3, 5, 100)
    plotSub(x, NormalDistribution(x,delta=1,u=0), ax,label='Normal distribution')
    
    y = np.linspace(0, 0.9, 30)
    x = np.zeros((len(y),))
    plotSub(x, y, ax,label='',linestyle='dotted')
    plt.savefig(imgSavePath+'dsitribution{}.png'.format(i)),plt.show()
      
def testTruncated_normal_distribution(i):
    ax = plt.subplot(1,1,1)
    plt.title('Truncated_normal_distribution')
    x = np.linspace(-10, 10, 100)
    plotSub(x, Truncated_normal_distribution(x), ax,label='u=-8,deta=2')
    plotSub(x, Truncated_normal_distribution(x,a=-10,b=10,u=0,deta=2), ax,label='u=0,deta=2')
    plotSub(x, Truncated_normal_distribution(x,a=-10,b=10,u=9,deta=9), ax,label='u=9,deta=9')
    plotSub(x, Truncated_normal_distribution(x,a=-10,b=10,u=0,deta=9), ax,label='u=0,deta=9')
    plt.savefig(imgSavePath+'dsitribution{}.png'.format(i)),plt.show()
    
def testDagum_distribution(i):
    ax = plt.subplot(1,1,1)
    plt.title('Dagum_distribution')
    x = np.linspace(0, 3, 100)
    plotSub(x, Dagum_distribution(x), ax,label='b=1,p=1,a=0.5')
    plotSub(x, Dagum_distribution(x,b=1,p=1,a=1), ax,label='b=1,p=1,a=1')
    plotSub(x, Dagum_distribution(x,b=1,p=1,a=2), ax,label='b=1,p=1,a=2')
    plotSub(x, Dagum_distribution(x,b=1,p=1,a=3), ax,label='b=1,p=1,a=3')
    plotSub(x, Dagum_distribution(x,b=1,p=1,a=4), ax,label='b=1,p=1,a=4')
    plt.savefig(imgSavePath+'dsitribution{}.png'.format(i)),plt.show()
    
def testStable_distribution(i):
    ax = plt.subplot(1,1,1)
    plt.title('Stable_distribution')
    x = np.linspace(-1, 1, 100)
    plotSub(x, Stable_distribution(x), ax,label='beta=0,c=1,alpha=0.5')
    plotSub(x, Stable_distribution(x,beta=0.25,c=1,alpha=0.5), ax,label='beta=0.25,c=1,alpha=0.5')
    plotSub(x, Stable_distribution(x,beta=0.50,c=1,alpha=0.5), ax,label='beta=0.50,c=1,alpha=0.5')
    plotSub(x, Stable_distribution(x,beta=0.75,c=1,alpha=0.5), ax,label='beta=0.75,c=1,alpha=0.5')
    plotSub(x, Stable_distribution(x,beta=1.0,c=1,alpha=0.5), ax,label='beta=1.0,c=1,alpha=0.5')
    plt.savefig(imgSavePath+'dsitribution{}.png'.format(i)),plt.show()
    
def testSkew_normal_distribution(i):
    ax = plt.subplot(1,1,1)
    plt.title('Skew_normal_distribution')
    x = np.linspace(-2, 2, 100)
    plotSub(x, Skew_normal_distribution(x), ax,label='alpha=-4')
    plotSub(x, Skew_normal_distribution(x,alpha=-1), ax,label='alpha=-1')
    plotSub(x, Skew_normal_distribution(x,alpha=0), ax,label='alpha=0')
    plotSub(x, Skew_normal_distribution(x,alpha=1), ax,label='alpha=1')
    plotSub(x, Skew_normal_distribution(x,alpha=4), ax,label='alpha=4')
    plt.savefig(imgSavePath+'dsitribution{}.png'.format(i)),plt.show()
    
def plotAllDistributions():
    ds = []
    ds.append(testDiscrete_uniform_distribution)    #Discrete distribution
    ds.append(testBinomial_distribution)
    ds.append(testPoisson_distribution)
    ds.append(testGeometric_distribution)
    ds.append(testHypergeometric_distribution)
    ds.append(testZipfsLaw)
    ds.append(testBeta_binomial_distribution)
    ds.append(testLogarithmic_distribution)
    ds.append(testConway_Maxwell_Poisson_distribution)
    ds.append(testSkellam_distribution)
    ds.append(testYule_Simon_distribution)
    ds.append(testZeta_distribution)
    
    ds.append(testUniform_distribution) #Continuous distribution
    ds.append(testNormalDistribution)
    ds.append(testCauchy)
    ds.append(testLaplace_distribution)
    ds.append(testGeneralized_logistic_distribution)
    ds.append(testGumbel_distribution)
    ds.append(testLogistic_distribution)
    ds.append(testLog_normal_distribution)
    ds.append(testWeibull_distribution)
    ds.append(testPareto_distribution)
    ds.append(testRayleigh_distribution)
    ds.append(testGamma_distribution)
    ds.append(testStudentT_distribution)
    ds.append(testBeta_distribution)
    ds.append(testChi_distribution)
    ds.append(testErlang_distribution)
    ds.append(testBoltzmann_distribution)
    ds.append(testVon_Mises_distribution)
    ds.append(testLogit_normal_distribution)
    ds.append(testIrwin_Hall_distribution)
    ds.append(testBates_distribution)
    ds.append(testKumaraswamy_distribution)
    ds.append(testReciprocal_distribution)
    ds.append(testTriangular_distribution)
    ds.append(testTrapezoidal_distribution)
    ds.append(testWigner_semicircle_distribution)
    ds.append(testF_distribution)
    ds.append(testLandau_distribution)
    ds.append(testJohnsonSU_distribution)
    ds.append(testShifted_Gompertz_distribution)
    ds.append(testLevy_distribution)
    ds.append(testNakagami_distribution)
    ds.append(testGompertz_distribution)
    ds.append(testFrechet_distribution)
    #ds.append(testFolded_normal_distribution)
    ds.append(testHalf_normal_distribution)
    ds.append(testTruncated_normal_distribution)
    ds.append(testDagum_distribution)
    ds.append(testStable_distribution)
    ds.append(testSkew_normal_distribution)
    
    for i,f in enumerate(ds):
        f(i)

def main():
    #plotAllDistributions()
    testBeta_distribution(32)
    pass

if __name__ == '__main__':
    main()

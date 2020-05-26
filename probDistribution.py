#Python3 Steven
#common proability distrubution
import numpy as np
import matplotlib.pyplot as plt

def plot(x,y):
    plt.plot(x,y)
    plt.show()

def plotSub(x,y,ax=None, aspect=False, label=''):
    ax.plot(x,y,label=label)
    #ax.title.set_text(name)
    if aspect:
        ax.set_aspect(1)
    ax.legend()
     
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
 
def StudentT_distribution(x):#https://en.wikipedia.org/wiki/Student%27s_t-distribution#Monte_Carlo_sampling
    pass

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

def testNormalD():
    x = np.linspace(-5.0, 5.0, 100)
    #y = NormalDistribution_pdf(x)
    #plot(x,y)
    
    ax = plt.subplot(1,1,1)
    plt.title('NormalDistribution')
    plotSub(x, NormalDistribution_pdf(x), ax,label='Normal')
    plotSub(x, NormalDistribution_pdf(x,u=-1,delta=0.5), ax,label='u=-1,delta=0.5')
    plotSub(x, NormalDistribution_pdf(x,u=1,delta=2), ax,label='u=1,delta=2')
    plt.show()
    
def testCauchy():
    x = np.linspace(-5.0, 5.0, 100)  
    ax = plt.subplot(1,1,1)
    plt.title('Cauchy')
    #plotSub(x, Cauchy_pdf(x), ax,label='Cauchy')
    plotSub(x, Cauchy_pdf(x,x0=0,scaler=0.75), ax,label='x0=0,scaler=1')
    plotSub(x, Cauchy_pdf(x,x0=0,scaler=1), ax,label='x0=0,scaler=1')
    #plotSub(x, NormalDistribution_pdf(x), ax,label='Normal')
    plotSub(x, Cauchy_pdf(x,x0=0,scaler=2), ax,label='x0=0,scaler=2')
    plotSub(x, Cauchy_pdf(x,x0=-2,scaler=1), ax,label='x0=-2,scaler=1')
    plt.show()
  
def testLaplace_distribution():
    x = np.linspace(-15.0, 15.0, 100)  
    ax = plt.subplot(1,1,1)
    plt.title('Laplace_distribution')
    plotSub(x, Laplace_distribution(x), ax,label='Laplace_distribution')
    plotSub(x, Laplace_distribution(x,u=0,b=2), ax,label='u=0,b=2')
    plotSub(x, Laplace_distribution(x,u=0,b=4), ax,label='u=0,b=4')
    plotSub(x, Laplace_distribution(x,u=-5,b=4), ax,label='u=5,b=4')
    plt.show()
 
def testLogistic_distribution():
    x = np.linspace(-5.0, 20.0, 100)  
    ax = plt.subplot(1,1,1)
    plt.title('Logistic_distribution')
    plotSub(x, Logistic_distribution(x), ax,label='Logistic_distribution')
    plotSub(x, Logistic_distribution(x,u=5,s=2), ax,label='u=5,b=2')
    plotSub(x, Logistic_distribution(x,u=9,s=3), ax,label='u=9,b=3')
    plotSub(x, Logistic_distribution(x,u=9,s=4), ax,label='u=9,b=4')
    plotSub(x, Logistic_distribution(x,u=6,s=2), ax,label='u=6,b=2')
    plotSub(x, Logistic_distribution(x,u=2,s=1), ax,label='u=2,s=1')
    plt.show()
    
def testLog_normal_distribution():
    x = np.linspace(0, 3.0, 100)  
    ax = plt.subplot(1,1,1)
    plt.title('Log_normal_distribution')
    plotSub(x, Log_normal_distribution(x), ax,label='Log_normal_distribution')
    plotSub(x, Log_normal_distribution(x,delta=0.25), ax,label='delta=0.25')
    plotSub(x, Log_normal_distribution(x,delta=0.5), ax,label='delta=0.5')
    #plotSub(x, Log_normal_distribution(x,delta=1.25), ax,label='delta=1.25')
    plt.show()

def testWeibull_distribution():
    x = np.linspace(0, 2.5, 100)  
    ax = plt.subplot(1,1,1)
    plt.title('Weibull_distribution')
    plotSub(x, Weibull_distribution(x,lamda=1,k=0.5), ax,label='lamda=1,k=0.5')
    plotSub(x, Weibull_distribution(x,lamda=1,k=1), ax,label='lamda=1,k=1')
    plotSub(x, Weibull_distribution(x,lamda=1,k=1.5), ax,label='lamda=1,k=1.5')
    plotSub(x, Weibull_distribution(x,lamda=1,k=5), ax,label='lamda=1,k=5')
    plt.show()  

def testPareto_distribution():
    x = np.linspace(0, 5, 100)  
    ax = plt.subplot(1,1,1)
    plt.title('Pareto_distribution')
    plotSub(x, Pareto_distribution(x,alpha=1), ax,label='alpha=1')
    plotSub(x, Pareto_distribution(x,alpha=2), ax,label='alpha=2')
    plotSub(x, Pareto_distribution(x,alpha=3), ax,label='alpha=3')
    plotSub(x, Pareto_distribution(x,alpha=1,Xm=2), ax,label='alpha=1,Xm=2')
    plt.show()  

def test():
    x = np.linspace(-5.0, 5.0, 100)  
    ax = plt.subplot(1,1,1)
    plt.title('Generalized_logistic_distribution')
    plotSub(x, Generalized_logistic_distribution(x), ax,label='GLD')
    plotSub(x, Generalized_logistic_distribution(x,alpha=0.5), ax,label='GLD alpha=0.5')
    plotSub(x, Gumbel_distribution(x),ax,label='Gumbel_distribution')
    plt.show()
    
def main():
    # testNormalD()
    #testCauchy()
    # testLaplace_distribution()
    # test()
    # testLogistic_distribution()
    #testLog_normal_distribution()
    # testWeibull_distribution()
    #testPareto_distribution()
    pass
    
if __name__ == '__main__':
    main()

#steven 21/03/2020
#01/04/2020  add derivative2
#common and interesting math fuction 

import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import cm
from activationFunc import *
import math
from scipy.special import gamma,factorial,beta
from scipy import integrate

def derivative(f,x,h=0.0001): #one parameter fuction derivative
    return (f(x+h)-f(x))/h

def derivative2(f,x,h=0.0001): #Second Derivative
    return (derivative(f,x+h,h) - derivative(f,x,h))/h

def normalDistribution(x):
    return 1/np.sqrt(2*np.pi) * np.exp(-0.5*x**2)
    
def softmaxFuc(x):
    softmax = np.exp(x)/np.sum(np.exp(x))
    #print(softmax)
    #print(np.sum(softmax))
    return softmax

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def log(x):
    return np.log(x)

def entropy(x):
    return x*np.log(x)
    
def exp(x):
    return np.exp(x)
    
def circle(x):
    return np.sqrt(1-x**2)*randomSymbol(len(x))

def heart(x): #heart equation: x**2+ (5*y/4 - sqrt(abs(x)))**2 = 1
    a = np.sqrt(1-x**2)*randomSymbol(len(x)) + np.sqrt(abs(x))
    return a*4/5 

def sawtoothWave(x): # y = t-floor(t) or y = t%1
    #return x%1
    #return x%1 + x
    #return x%1 + np.sin(x)*x
    a = 10 #perid
    return 2*(x/a - np.floor(0.5+x/a))

def powerX(x):
    #return np.power(x,x)
    return powerAX(x,x)

def powerAX(a,x):
    return np.power(a,x)

def logisticMap2(r=1.5,x0=0.8,N=10): #x:= r*x(1-x)
    maps=[]
    a = x0
    while N>0:
        #print(a,' ',end='')
        maps.append(a)
        a = r*a*(1-a)**3
        N -= 1

    return maps

def logisticMap(r=1.5,x0=0.8,N=10): #x:= r*x(1-x)
    maps=[]
    a = x0
    while N>0:
        #print(a,' ',end='')
        maps.append(a)
        a = r*a*(1-a)
        N -= 1

    return maps

def randomSymbol(length): #-1,1,1,-1,1,-1...
    f = np.zeros(length)+np.where(np.random.random(length)>0.5,1.0,-1.0)
    return f

def Divisorfunction(param): #https://en.wikipedia.org/wiki/Divisor_function#Definition
    def getDivsorList(N):
        for i in range(1,N+1):
            #print('i=',i)
            if N%i == 0:
                yield i
            
    N = param[0]
    p = param[1]
    def powerF(x):
        #print('x=',x,np.power(x,p))
        return np.power(x,p)
    
    l = list(map(powerF, [i for i in getDivsorList(N)]))
    return np.sum(l)

    
def EulerTotients(N=10):#https://en.wikipedia.org/wiki/Euler%27s_totient_function
    def gcd(p,q): # Create the gcd of two positive integers.
        while q != 0:
            p, q = q, p%q
        return p

    def is_coprime(x, y):
        return gcd(x, y) == 1

    def Totients(x):
        if x == 1:
            return 1
        n = [y for y in range(1,x) if is_coprime(x,y)]
        #print('n=',n)
        return len(n)

    return Totients(N)

def PrimeNumbers(N=10): #https://en.wikipedia.org/wiki/Prime-counting_function
    def isPrime(num):
        if num>1:            
            for n in range(2, num): 
                if (num % n) == 0: 
                    return False
            return True
        else:
            return False
                            
    def getPrime():
        for i in range(2,N+1):
            if isPrime(i):
                yield i
            
    l = [i for i in getPrime()]
    print('l=',len(l),l)
    return len(l)

def Mobiusfunction(N=10): #https://en.wikipedia.org/wiki/M%C3%B6bius_function
    def IsSquareFreeIntegers(x):
        for n in range(2, int(np.sqrt(x)+1)): 
            if (x % n**2) == 0: 
                return False
        return True
        
    def SquareFreeIntegers2():
        squareFreeNumbers=[]
        for val in range(1, N + 1):
            if IsSquareFreeIntegers(val):
                squareFreeNumbers.append(val)
        return squareFreeNumbers
            
    def SquareFreeIntegers():        
        bSqure=False
        for val in range(1, N + 1): 
            for n in range(2, int(np.sqrt(N)+1)): 
                bSqure=False
                if (val % n**2) == 0: 
                    bSqure=True
                    break
            if not bSqure:
                yield val
                
    def Mobius(x):
        if IsSquareFreeIntegers(x):
            nPrime = PrimeNumbers(x)
            print('x,nPrime=',x,nPrime)
            if nPrime % 2 ==0:
                return 1
            else:
                return -1
        else:
            return 0
        
    #squareFreeNumbers = SquareFreeIntegers2() #[i for i in SquareFreeIntegers()]
    #print('squareFreeNumbers=',len(squareFreeNumbers),squareFreeNumbers)
    #l = list(map(Mobius, [i for i in range(N)]))
    #print('l=',len(l),l)
    return Mobius(N)
     
def LegendreFunction(x,n=0): #https://en.wikipedia.org/wiki/Legendre_function
    if n == 0:
        return 0.5*np.log((1+x)/(1-x))
    elif n == 1:
        return x*LegendreFunction(x,0) -1 #P1(x) = x
    else:
        return ((2*n-1)/n)*x*LegendreFunction(x,n-1) - ((n-1)/n)*LegendreFunction(x,n-2)
    
def ScorersFunctionGi(x): #https://en.wikipedia.org/wiki/Scorer%27s_function
    def gi(t):
        return (1/np.pi)*np.sin((1/3)*t**3 + x*t)
    # return integrate.quad(gi, 0, 1)[0]
    def hi(t):
        return (1/np.pi)*np.exp((-1/3)*t**3 + x*t)
    
    # rGi = integrate.quad(gi, 0, np.inf,epsabs=1.49e-8, 
    #                epsrel=1.49e-8, maxp1=50, limit=50)[0]
    # rHi = integrate.quad(hi, 0, np.inf,epsabs=1.49e-8, 
    #                epsrel=1.49e-8, maxp1=50, limit=50)[0]
    # return rGi,rHi
    return integrate.quad(gi, 0, np.inf)[0], integrate.quad(hi, 0, np.inf)[0]
    

#------------------start plot------------------------------
def plot(x,y=None):
    if y:
        plt.plot(x,y)
    else:
        plt.plot(x)
    plt.show()

def plotSub(x,y,ax=None, aspect=False, label=''):
    ax.plot(x,y,label=label)
    #ax.title.set_text(title)
    if aspect:
        ax.set_aspect(1)
    ax.legend()
    
def scatterSub(x,y,ax=None,label='',marker='.',c='r'):
    ax.scatter(x,y,linewidths=.3,c=c,label=label,marker=marker)
    #ax.set_aspect(1)

def scatter(x,y,ratio=True):
    plt.scatter(x,y)
    if ratio:
        ax = plt.gca()
        ax.set_aspect(1)
    plt.show()

def testLogisticMap():
    rValue = 3.9
    ls1 = logisticMap(r=rValue,x0=0.2,N=100)
    #plot(ls1)
    ls2 = logisticMap(r=rValue,x0=0.20000000001,N=100)
    #plot(ls2)

    ax = plt.subplot(1, 1, 1)
    ax.plot(ls1,label='x0=0.2')
    ax.plot(ls2,label='x0=0.20000000001')

    xMax = 100
    ax.set_xticks(np.linspace(0,xMax,10))
    ax.set_xlim(0, xMax)
    ax.set_title("An iota causes a big difference(butterfly effects)")
    ax.legend(loc='lower left')
    plt.show()

def testLogisticMap2():
    #rValue = 9
    #ls1 = logisticMap2(r=rValue,x0=0.2,N=100)
    #plot(ls1)
    #ls2 = logisticMap(r=rValue,x0=0.20000000001,N=100)
    #plot(ls2)

    #x = np.linspace(-1,1, 10000)
    #scatter(x,circle(x))
    #scatter(x,heart(x))
    #Num=100
    #scatter(np.arange(Num),logisticMap(r=.5,x0=0.4,N=Num),False)
    #plot(logisticMap(r=.5,x0=0.4,N=30))
    
    '''
    N=9
    for i in range(N):
        print('*'*50,i)
        rValue = i+1
        ls1 = logisticMap2(r=rValue,x0=0.2,N=100)
        
        ax = plt.subplot(3, 3, i+1)
        ax.plot(ls1,label='x0=0.2')
        ax.set_title('r='+str(rValue))
    plt.show()
    '''

    iters=80
    rStart = 3.9
    #rl=[0, 0.01, 0.001, 0.0001, 0.0000001, 0.0000000001]
    rl=[0, 0.000001, 0.0000000000001]
    ls=[]
    for i,rV in enumerate(rl):
        #print('*'*50,i)
        rValue = rStart + rV
        ls.append(logisticMap(r=rValue,x0=0.8,N=iters))
        
    ax = plt.subplot(1,1,1)
    for i,l in enumerate(ls):
        ax.plot(l,label='x0='+str(rStart + rl[i]))

    ax.set_title('Butterfly effects')
    ax.legend(loc='lower left')
    plt.show()

def plotCircle(ax):
    x = np.linspace(-1.0, 1.0, 100)
    y = np.linspace(-1.0, 1.0, 100)
    X, Y = np.meshgrid(x, y)
    F = X ** 2 + Y ** 2 - 1
    ax.contour(X, Y, F, [0])
    ax.set_aspect(1)
    
def plotHeart(ax):
    x = np.linspace(-1.0, 1.0, 100)
    y = np.linspace(-1.0, 1.5, 100)
    X, Y = np.meshgrid(x, y)
    # x**2+ (5*y/4 - sqrt(abs(x)))**2 = 1
    F = X ** 2 + (5*Y/4 - np.sqrt(abs(x)))** 2 - 1
    ax.contour(X, Y, F, [0])
    ax.set_aspect(1)

def plotPowerX():
    x = np.linspace(0,1.5, 100)
    ax = plt.subplot(1,1,1)
    plotSub(x, powerX(x), ax,label='powerX')
    #plotSub(x, derivative(powerX, x), ax, label='powerX\'', aspect=True)
    
    de = derivative(powerX, x)
    de[x<0.1]= np.nan
    plotSub(x, de, ax, label='powerX\'', aspect=False)
    
    #plotSub(x, derivative2(powerX, x), ax, label='powerX\'\'', aspect=True)
    plotSub(x, x, ax,label='y=x')
    plotSub(x, np.ones((len(x))), ax,label='y=1')
    plt.show()

def plotAllFuc():
    x = np.linspace(-2,2, 50)
    ax = plt.subplot(1,1,1)
    
    plotSub(x, sigmoid(x), ax,label='sigmoid')
    plotSub(x, derivative(sigmoid, x), ax, label='sigmoid\'', aspect=False)
    plotSub(x, derivative2(sigmoid, x), ax, label='sigmoid\'\'', aspect=False)
    
    plotSub(x, normalDistribution(x), ax,label='normalDistribution')
    plotSub(x, derivative(normalDistribution, x), ax, label='normalDistribution\'', aspect=False)
    plotSub(x, derivative2(normalDistribution, x), ax, label='normalDistribution\'\'', aspect=False)
    
    #plotSub(x, exp(x), ax,name='exp', label='exp')
    #plotSub(x, derivative(exp, x), ax, label='exp\'', aspect=False)
    
    plotSub(x, softmaxFuc(x), ax,label='softmaxFuc')
    plotSub(x, derivative(softmaxFuc, x), ax, label='softmaxFuc\'', aspect=False)
    
    #plotSub(x, log(x), ax,label='log') #log(x)' = 1/x
    #plotSub(x, derivative(log, x), ax,label='log\'')
    
    plotHeart(ax)
    plotCircle(ax)
    plt.show()
    
def plotSoftmax():
    x = np.linspace(-2,-1, 100)
    ax = plt.subplot(1,1,1)
    plotSub(x, softmaxFuc(x), ax,label='softmaxFuc[-2,-1]')
    
    x = np.linspace(-1,0, 100)
    plotSub(x, softmaxFuc(x), ax,label='softmaxFuc[-1,0]')
    
    x = np.linspace(-0.5,0.5, 100)
    plotSub(x, softmaxFuc(x), ax,label='softmaxFuc[-0.5,0.5]')
    
    x = np.linspace(0,1, 100)
    plotSub(x, softmaxFuc(x), ax,label='softmaxFuc[0,1]')
    
    x = np.linspace(1,3, 100)
    plotSub(x, softmaxFuc(x), ax,label='softmaxFuc[1,3]')
    
    x = np.linspace(3,5, 100)
    plotSub(x, softmaxFuc(x), ax,label='softmaxFuc[3,5]')
    
    x = np.linspace(-1,3, 100)
    plotSub(x, softmaxFuc(x), ax,label='softmaxFuc[-1,3]')
    
    x = np.linspace(-2,5, 100)
    plotSub(x, softmaxFuc(x), ax,label='softmaxFuc[-2,5]')
    
    plt.show()

def plotSoftmax2():
    ax = plt.subplot(1,1,1)
    Ns=[10,20,50,100,1000]    
    for i in Ns:
        x = np.linspace(0,1, i)
        plotSub(x, softmaxFuc(x), ax,label='softmaxFuc[0,1]-'+str(i))
    plt.show()
    
def plotActivationFucBRelu():
    ax = plt.subplot(1,1,1)
    x = np.linspace(-0.1,0.1, 40)
    ax.set_title('Activation BRelu')
    plotSub(x, BReLu(x), ax,label='BReLu-even')
    y = np.linspace(-0.1,0.1, 41)
    plotSub(y, BReLu(y), ax,label='BReLu-odd')
    plt.show()
    
def plotActivationOneFun():
    ax = plt.subplot(1,1,1)
    x = np.linspace(-20,20, 100)   
    ax.set_title('Activation')
    
    #plotSub(x, softPlus(x), ax,label='softPlus')
    #plotSub(x, BentIdentity(x), ax,label='BentIdentity')
    #plotSub(x, SoftClipping(x), ax,label='SoftClipping')
    
    #plotSub(x, SoftExponential(x), ax,label='SoftExponential')
    #plotSub(x, Sinusoid(x), ax,label='Sinusoid')
    #plotSub(x, Sinc(x), ax,label='Sinc')
    #plotSub(x, Gaussian(x), ax,label='Gaussian')
    
    x = np.linspace(-5,5, 100)   
    plotSub(x, SQ_RBF(x), ax,label='SQ_RBF')
    plt.show()

def plotEquationSlove():
    ax = plt.subplot(1,1,1)
    x = np.linspace(-4,5, 100)   
    ax.set_title('x^2 = 2^x, not just 2 and 4')
    plotSub(x, x**2, ax,label='x^2')
    plotSub(x, powerAX(2,x), ax,label='2^x')
    plt.show()
    
def plotActivationFunSoftE():
    ax = plt.subplot(1,1,1)
    x = np.linspace(-2,2, 50)   
    ax.set_title('Activation SoftExponential')
    
    alphas = np.linspace(-1,1, 9)
    for i in alphas:
        plotSub(x, SoftExponential(x,i), ax,label='SoftExponential_alpha('+str(i)+')')
    plt.show()

def plotActivationFunSquashing():
    ax = plt.subplot(1,1,1)
    x = np.linspace(-2,2, 100)   
    ax.set_title('Activation Squashing')
    plotSub(x, Squashing(x,beta=1), ax,label='Squashing_beta=1')
    plotSub(x, Squashing(x,beta=2), ax,label='Squashing_beta=2')
    plotSub(x, Squashing(x,beta=5), ax,label='Squashing_beta=5')
    plotSub(x, Squashing(x,beta=50), ax,label='Squashing_beta=50')
    plt.show()
    
def plotActivationFun():
    ax = plt.subplot(1,1,1)
    
    x = np.linspace(-2,2, 80)   
    ax.set_title('Activation Function')
    
    plotSub(x, Identity(x), ax,label='Identity')
    plotSub(x, Binary_step(x), ax,label='Binary_step')
    plotSub(x, sigmoid(x), ax,label='sigmoid')
    plotSub(x, Tanh(x), ax,label='Tanh')
    plotSub(x, SQNL(x), ax,label='SQNL')
    plotSub(x, ArcTan(x), ax,label='ArcTan')
    plotSub(x, ArcSinH(x), ax,label='ArcSinH')
    
    plotSub(x, Softsign(x), ax,label='Softsign')
    plotSub(x, ISRu(x), ax,label='ISRu')
    plotSub(x, ISRLu(x), ax,label='ISRLu')
    plotSub(x, PLu(x), ax,label='PLu')
    plotSub(x, Relu(x), ax,label='Relu')
    
    plotSub(x, LeakyRelu(x), ax,label='LeakyRelu')
    plotSub(x, PRelu(x), ax,label='PRelu')
    plotSub(x, RRelu(x), ax,label='RRelu')
    plotSub(x, GELu(x), ax,label='GELu')
    plotSub(x, ELu(x), ax,label='ELu')
    plotSub(x, SELu(x), ax,label='SELu')
    
    plotSub(x, BReLu(x), ax,label='BReLu-even')
    y = np.linspace(-2,2, 51)   
    plotSub(y, BReLu(y), ax,label='BReLu-odd')

    plotSub(x, softPlus(x), ax,label='softPlus')
    plotSub(x, BentIdentity(x), ax,label='BentIdentity')
    plotSub(x, SoftClipping(x), ax,label='SoftClipping')
    plotSub(x, Squashing(x,beta=1), ax,label='Squashing_beta=1')
    plotSub(x, SoftExponential(x), ax,label='SoftExponential')
    plotSub(x, Sinusoid(x), ax,label='Sinusoid')
    plotSub(x, Sinc(x), ax,label='Sinc')
    plotSub(x, Gaussian(x), ax,label='Gaussian')
    plotSub(x, SQ_RBF(x), ax,label='SQ_RBF')
    
    plt.ylim(-2, 6)
    #plt.legend()
    plt.legend(ncol=4,loc='upper left')    
    plt.show()
    
def plotLogEntropy():
    ax = plt.subplot(1,1,1)
    x = np.linspace(0,4, 50)   
    ax.set_title('LogEntropy Function')
    
    plotSub(x, log(x), ax,label='log')
    plotSub(x, entropy(x), ax,label='entropy')
    plt.legend()
    plt.show()
   
def plot_gamma():
    ax = plt.subplot(1,1,1)
    x = np.linspace(-10,7, 1000)   
    ax.set_title('gamma Function')
    plotSub(x, gamma(x), ax,label='gamma(x)')
    
    k = np.arange(1, 8)
    scatterSub(k, factorial(k-1),ax, label='(x-1)!, x = 1, 2, ...', marker='*')
    
    plt.xlim(-7.5, 8.5)
    plt.ylim(-40, 160)
    plt.grid()
    plt.xlabel('x')
    plt.legend(loc='lower right')
    plt.show()

def yieldBetaFuc(start,stop,N):  
    xvalues = np.linspace(start,stop,N)
    yvalues = np.linspace(start,stop,N)
    for u, x in enumerate(xvalues):
        for v, y in enumerate(yvalues):
            #print('x,y=',x,y,beta(x,y))
            yield u,v,beta(x,y)

def getBetaFucImg(start,stop,N):
    M = np.zeros([N, N,3], int) # + 255
    for v,u,z in yieldBetaFuc(start,stop,N): #map z(0~2) to 0~255 pixsel value
        value =  int(z*256/.2)
        #print('z=',v,u,z,value)
        #M[v, u, :] = value
        #M[v, u, 0] = value #r channel     
        M[v, u, 1] = value #g channel
        #M[v, u, 2] = value #b channel             
    return M

def plotBetaFuc2():
    start=1
    stop =3
    N=1000
    x = np.linspace(start,stop, N)  
    plt.imshow(getBetaFucImg(start,stop,N),cmap='gray')
    plt.show()
    
def plot_beta():
    ax = plt.subplot(1,1,1)
    x = np.linspace(-3,3, 200)  
    print(x.shape)
    x,y = np.meshgrid(x,x)
    v = beta(x,y).flatten()
    print(v.shape)
    print(x.shape)
    print(y.shape)
    #return
    ax.set_title('beta Function')
    #plotSub(x, y, ax,label='beta(x)')
    #scatterSub(x,beta(x,y),ax, label='beta', marker='.')
    scatterSub(x,y,ax, c=v)
    
    #plt.grid()
    plt.xlabel('x')
    plt.legend(loc='lower right')
    plt.show()

def plotHeart():
    row = 2
    col = 2
    ax = plt.subplot(row, col, 1)
    x = np.linspace(-5,5, 10)
    plotSub(x,sigmoid(x), ax,label='sigmod')

    ax = plt.subplot(row, col, 2)
    x = np.linspace(-5,0, 10)
    plotSub(x,exp(x), ax,label='exp')

    ax = plt.subplot(row, col, 3)
    x = np.linspace(-1,1, 1000)
    scatterSub(x,circle(x), ax,label='circle')

    ax = plt.subplot(row, col, 4)
    scatterSub(x,heart(x), ax,label='heart')
    plt.show()
    
def plotDivisorfunction(N=200):
    ax = plt.subplot(1,1,1)
    ax.set_title('Divisor function')
    
    x = np.arange(1,N)
    y = list(map(Divisorfunction, [(i,0) for i in x]))
    plotSub(x, y, ax,label='N=250,p=0')
    
    y1 = list(map(Divisorfunction, [(i,1) for i in x]))
    plotSub(x, y1, ax,label='N=250,p=1')
    
    #y2 = list(map(Divisorfunction, [(i,2) for i in x]))
    #plotSub(x, y2, ax,label='N=250,p=2')
    plt.show()
    
def plotEulerTotients():
    ax = plt.subplot(1,1,1)
    ax.set_title('Euler Totients')
    N=500
    x = np.arange(1,N)
    y = list(map(EulerTotients, [i for i in x]))
    #print('y=',y)
    scatterSub(x,y,ax,label='totients')
    plt.show()
    
def plotPrimeNumbers():
    ax = plt.subplot(1,1,1)
    ax.set_title('Prime counter Function')
    
    N=60
    x = np.arange(1,N)
    y = list(map(PrimeNumbers, [i for i in x]))
    scatterSub(x,y,ax,label='Prime counter')
    
    x = np.linspace(0,N, 100)
    y = x/np.log(x)
    plotSub(x, y, ax,label='x/ln(x)')
    
    ax.legend()
    plt.show()
    
def plotMobiusfunction():
    ax = plt.subplot(1,1,1)
    ax.set_title('Mobius function')
    
    N=60
    x = np.arange(1,N)
    y = list(map(Mobiusfunction, [i for i in x]))
    print(x)
    print(y)
    scatterSub(x,y,ax,label='Mobius')
    plt.show()
    
def plotLegendreFunction():
    ax = plt.subplot(1,1,1)
    ax.set_title('Legendre Function')
    
    x = np.linspace(-1,1, 200)
    plotSub(x, LegendreFunction(x), ax,label='Q0')
    plotSub(x, LegendreFunction(x,n=1), ax,label='Q1')
    plotSub(x, LegendreFunction(x,n=2), ax,label='Q2')
    plotSub(x, LegendreFunction(x,n=3), ax,label='Q3')
    plotSub(x, LegendreFunction(x,n=4), ax,label='Q4')
    ax.legend()
    plt.show()
    
def plotScorersFunction():
    ax = plt.subplot(1,1,1)
    ax.set_title('Scorers Function')
    
    x = np.linspace(-10,8, 50)
    yGi=[]
    yHi=[]
    for _ in x:
        gi,hi =ScorersFunctionGi(_)
        yGi.append(gi)
        yHi.append(hi)
        
    plotSub(x, yGi, ax,label='Gi')
    plotSub(x, yHi, ax,label='Hi')
    #plotSub(x, LegendreFunction(x,n=1), ax,label='Q1')
    
    ax.legend()
    plt.show()
    
def main():
    #return testLogisticMap2()    
    #plotAllFuc()
    #plotPowerX()
    #plotSoftmax()
    #plotSoftmax2()
    #plotActivationFun()
    #plotActivationFunSoftE()
    #plotActivationFucBRelu()
    #plotActivationFunSquashing()
    #plotActivationOneFun()
    #plotEquationSlove()
    #plotLogEntropy()
    #plot_gamma()
    #plot_beta()
    #plotBetaFuc2()
    #plotDivisorfunction()
    #plotHeart()
    #plotEulerTotients()
    #plotPrimeNumbers()
    #plotMobiusfunction()
    #plotLegendreFunction()
    plotScorersFunction()
    pass

if __name__ == '__main__':
    main()

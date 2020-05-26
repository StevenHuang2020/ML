#steven 21/03/2020
#01/04/2020  add derivative2
#common and interesting math fuction 

import numpy as np 
import matplotlib.pyplot as plt
from activationFunc import *
import math
from scipy.special import gamma,factorial,beta

def derivative(f,x,h=0.0001): #one parameter fuction derivate
    return (f(x+h)-f(x))/h

def derivative2(f,x,h=0.001): #Second Derivative
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

def plot(x,y=None):
    if y:
        plt.plot(x,y)
    else:
        plt.plot(x)
    plt.show()

def plotSub(x,y,ax=None, aspect=False, label=''):
    ax.plot(x,y,label=label)
    #ax.title.set_text(name)
    if aspect:
        ax.set_aspect(1)
    ax.legend()
    
def scatterSub(x,y,ax=None,label='',marker=','):
    ax.scatter(x,y,linewidths=.3,color='r',label=label,marker=marker)
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
    x = np.linspace(-0.5,0.5, 20)
    ax.set_title('Activation BRelu')
    plotSub(x, BReLu(x), ax,label='BReLu-even')
    y = np.linspace(-0.5,0.5, 21)   
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
    
def plotActivationFun():
    ax = plt.subplot(1,1,1)
    
    x = np.linspace(-2,2, 50)   
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
    x = np.linspace(-3.5,5, 1000)   
    ax.set_title('gamma Function')
    plotSub(x, gamma(x), ax,label='gamma(x)')
    
    k = np.arange(1, 7)
    scatterSub(k, factorial(k-1),ax, label='(x-1)!, x = 1, 2, ...', marker='*')
    
    plt.xlim(-3.5, 5.5)
    plt.ylim(-10, 25)
    plt.grid()
    plt.xlabel('x')
    plt.legend(loc='lower right')
    plt.show()

def plot_beta():
    ax = plt.subplot(1,1,1)
    x = np.linspace(-3.5,5, 1000)   
    ax.set_title('beta Function')
    plotSub(x, gamma(x), ax,label='beta(x)')
    
    k = np.arange(1, 7)
    scatterSub(k, factorial(k-1),ax, label='(x-1)!, x = 1, 2, ...', marker='*')
    
    plt.xlim(-3.5, 5.5)
    plt.ylim(-10, 25)
    plt.grid()
    plt.xlabel('x')
    plt.legend(loc='lower right')
    plt.show()

def main():
    #return testLogisticMap2()
    
    #x = np.linspace(-1,1, 10000)
    #scatter(x,circle(x))
    #scatter(x,heart(x))
    #Num=100
    #scatter(np.arange(Num),logisticMap(r=.5,x0=0.4,N=Num),False)
    #plot(logisticMap(r=.5,x0=0.4,N=30))

    '''
    row = 2
    col = 2
    ax = plt.subplot(row, col, 1)
    x = np.linspace(-5,5, 10)
    plotSub(x,sigmoid(x), ax,name='sigmod')

    ax = plt.subplot(row, col, 2)
    x = np.linspace(-5,0, 10)
    plotSub(x,exp(x), ax,name='exp')

    ax = plt.subplot(row, col, 3)
    x = np.linspace(-1,1, 1000)
    scatterSub(x,circle(x), ax,name='circle')

    ax = plt.subplot(row, col, 4)
    scatterSub(x,heart(x), ax,name='heart')
    plt.show()
    '''

    '''
    x = np.linspace(0,1.5, 100)
    ax = plt.subplot(1,1,1)
    #plotSub(x,sawtoothWave(x),ax,label='sawWave')
    plotSub(x, powerX(x), ax,label='X^x')
    plotSub(x, derivative(powerX, x), ax, label='X^x\'', aspect=False)
    '''
    
    #x = np.linspace(0,2, 100)
    #plotSub(x, powerX(x), ax,label='powerX')
    #plotSub(x, derivative(powerX, x), ax, label='powerX\'', aspect=False)
    #plt.show()
    
    #plotAllFuc()
    #plotPowerX()
    #plotSoftmax()
    #plotSoftmax2()
    #plotActivationFun()
    #plotActivationFunSoftE()
    #plotActivationFucBRelu()
    #plotActivationOneFun()
    #plotEquationSlove()
    #plotLogEntropy()
    plot_gamma()
    pass

if __name__ == '__main__':
    main()

#steven 21/03/2020
#01/04/2020  add derivative2
#common and interesting math fuction 

import numpy as np 
import matplotlib.pyplot as plt


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
    return np.power(x,x)

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
    
def scatterSub(x,y,ax=None,name=''):
    ax.scatter(x,y,linewidths=.1,color='r',marker=',')
    ax.set_aspect(1)

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
    plotSoftmax2()
    pass

if __name__ == '__main__':
    main()

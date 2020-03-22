#steven 21/03/2020
#common and interesting math fuction 
from __future__ import unicode_literals

import numpy as np 
import matplotlib.pyplot as plt
import matplotlib


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

def randomSymbol(length): #-1,1,1,-1,1,-1...
    f = np.zeros(length)+np.where(np.random.random(length)>0.5,1.0,-1.0)
    return f
    
def circle(x):
    return np.sqrt(1-x**2)*randomSymbol(len(x))

def heart(x): #heart equation: x**2+ (5*y/4 - sqrt(abs(x)))**2 = 1
    a = np.sqrt(1-x**2)*randomSymbol(len(x)) + np.sqrt(abs(x))
    return a*4/5 

def logisticMap(r=1.5,x0=0.8,N=10): #x:= r*x(1-x)
    maps=[]
    a = x0
    while N>0:
        #print(a,' ',end='')
        maps.append(a)
        a = r*a*(1-a)
        N -= 1

    return maps

def plot(x,y=None):
    if y:
        plt.plot(x,y)
    else:
        plt.plot(x)
    
    plt.show()

def plotSub(x,y,ax=None,name=''):
    ax.plot(x,y)
    ax.title.set_text(name)
    #ax.set_aspect(1)

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
    ax.set_title("An iota causes a long distance")
    ax.legend(loc='lower left')
    plt.show()

def plotCircle():
    x = np.linspace(-1.0, 1.0, 100)
    y = np.linspace(-1.0, 1.0, 100)
    X, Y = np.meshgrid(x, y)
    F = X ** 2 + Y ** 2 - 1
    plt.contour(X, Y, F, [0])
    plt.show()

def plotHeart():
    x = np.linspace(-1.0, 1.0, 100)
    y = np.linspace(-1.0, 1.5, 100)
    X, Y = np.meshgrid(x, y)
    # x**2+ (5*y/4 - sqrt(abs(x)))**2 = 1
    F = X ** 2 + (5*Y/4 - np.sqrt(abs(x)))** 2 - 1
    plt.contour(X, Y, F, [0])
    plt.show()

def main():
    #return testLogisticMap()

    #x = np.linspace(-1,1, 10000)
    #scatter(x,circle(x))
    #scatter(x,heart(x))
    #Num=100
    #scatter(np.arange(Num),logisticMap(r=.5,x0=0.4,N=Num),False)
    #plot(logisticMap(r=.5,x0=0.4,N=30))

    #return plotCircle()
    #return plotHeart()

    ax = plt.subplot(2, 2, 1)
    x = np.linspace(-5,5, 10)
    plotSub(x,sigmoid(x), ax,name='sigmod')

    ax = plt.subplot(2, 2, 2)
    x = np.linspace(-5,0, 10)
    plotSub(x,exp(x), ax,name='exp')

    ax = plt.subplot(2, 2, 3)
    x = np.linspace(-1,1, 1000)
    scatterSub(x,circle(x), ax,name='circle')

    ax = plt.subplot(2, 2, 4)
    scatterSub(x,heart(x), ax,name='heart')

    plt.show()

if __name__ == '__main__':
    main()

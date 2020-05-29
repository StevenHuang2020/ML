#steven 01/03/2020
#calculate pi using random method. (Monte Carlo method)
import numpy as np 
import matplotlib.pyplot as plt
import sympy #symbol integrate calculate
from sympy import *

def randomSeries(N):#generate series number between 0~1
    return np.random.rand(N)
   
def distance(a, b):#euclidean distance from point(a,b) to point(0,0)"""
    return np.sqrt(a**2 + b**2)

def circleFun(x):
    return np.sqrt(1-x**2)

def plotXY(x,y):
    plt.figure(num='Calculate Pi')
    ax = plt.gca()
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    for a,b in zip(x,y):
        if distance(a, b) > 1:
            ax.scatter(a, b, c='r', s=5, alpha=0.5)
        else:
            ax.scatter(a, b, c='b', s=5, alpha=0.5)

    plt.show()

def menteCarloMethod(N=10000 ,plot=True):
    x = randomSeries(N)
    y = randomSeries(N)

    res = np.where(distance(x,y) > 1, 0, 1)
    #print(res)
    
    pi = np.sum(res)/len(res)*4.0
    #pi = np.mean(res == 1)*4.0
    print('when N=',N,'pi=',pi)
    
    if plot:
        plotXY(x,y) 

def IntegralCalculatePi(N=10000):
    """calculate area form x = 0 to 1
    divide 0~1 to N shares,  erevy part considered as a rectangle.
    """
    s = 0
    for i in range(N):
        x = 1/N
        y = circleFun(i/N)
        s += x*y
    
    pi = s*4 
    print('when N=',N,'s = ',s,'pi = ',pi)
    fillColor()

def fillColor():
    x = np.linspace(0, 1, 100)
    y1 = np.zeros(len(x))
    y2 = circleFun(x)

    plt.plot(x,y1,c='b',alpha=0.5)
    plt.plot(x,y2,c='b',alpha=0.5)
    
    plt.fill_between(x,y1,y2,where=x<=1,facecolor='green')
    #plt.axes(aspect='equal')#.set_aspect('equal')
    plt.axes().set_aspect('equal')
    #plt.grid(True)
    plt.show()

def IntegralAccPi():
    x = sympy.Symbol('x')
    y = sqrt(1-x**2)
    print(sympy.integrate(y, (x, 0, 1)))
    
def main():
    #menteCarloMethod(plot=False)
    IntegralCalculatePi()
    #IntegralAccPi()

if __name__ == "__main__":
    main()

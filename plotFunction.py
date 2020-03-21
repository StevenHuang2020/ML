#steven 21/03/2020
#common and interesting math fuction 
import numpy as np 
import matplotlib.pyplot as plt

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
    #return 4/5*np.sqrt((1-x**2)*abs(x))*randomSymbol(len(x))
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

    maps.reverse()
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
    ax.scatter(x,y,linewidths=.5)
    ax.set_aspect(1)

def scatter(x,y,ratio=True):
    plt.scatter(x,y)
    if ratio:
        ax = plt.gca()
        ax.set_aspect(1)
    plt.show()

def main():
    x = np.linspace(-1,1, 10000)
    #scatter(x,circle(x))
    #scatter(x,heart(x))
    Num=100
    #scatter(np.arange(Num),logisticMap(r=.5,x0=0.4,N=Num),False)
    #plot(logisticMap(r=.5,x0=0.4,N=30))

    ax = plt.subplot(2, 2, 1)
    x = np.linspace(-5,5, 10)
    plotSub(x,sigmoid(x), ax,name='sigmod')

    ax = plt.subplot(2, 2, 2)
    x = np.linspace(-5,0, 10)
    plotSub(x,exp(x), ax,name='exp')

    ax = plt.subplot(2, 2, 3)
    x = np.linspace(-1,1, 10000)
    scatterSub(x,circle(x), ax,name='circle')

    ax = plt.subplot(2, 2, 4)
    scatterSub(x,heart(x), ax,name='heart')

    plt.show()

if __name__ == '__main__':
    main()

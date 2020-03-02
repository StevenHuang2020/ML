#steven 01/03/2020
#calculate pi using random method. (Monte Carlo method)
import numpy as np 
import matplotlib.pyplot as plt

def randomSeries(N):
    """generate series number between 0~1"""
    return np.random.rand(N)
   
def distance(a, b):
    """calculate the distance from point(a,b) to point(0,0)"""
    return np.sqrt(a**2 + b**2)

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

def main():
    N = 30000  #samples
    x = randomSeries(N)
    y = randomSeries(N)

    #print(x)
    #print(y)
    print(distance(x,y))
    res = np.where(distance(x,y) > 1, 0, 1)
    #print(res)
    pi = np.sum(res)*4.0/len(res)
    print(pi)
    
    plotXY(x,y)

if __name__ == "__main__":
    main()

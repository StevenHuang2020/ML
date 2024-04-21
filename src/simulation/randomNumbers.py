#python3 Steven
import random
import numpy as np
from plotCommon import *

def basic(x0=0,N=100):
    res = [x0]
    for i in range(N):
        #res.append(res[-1] + random.random()/(i+1))
        res.append(res[-1] + i*np.random.randn())

    return res

def plotRandomNumbers(Time=100):
    N = 30
    ax = plt.subplot(1,1,1)
    plt.title('Random Numbers')
    x = np.arange(N+1)
    for i in range(Time):
        y = basic(N=N)
        plotSub(x,y,ax)
    plt.show()

def main():
    plotRandomNumbers()

if __name__=='__main__':
    main()

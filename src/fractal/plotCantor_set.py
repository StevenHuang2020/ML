#python3
#steven 05/04/2020 Cantor set
#https://en.wikipedia.org/wiki/Cantor_set
import matplotlib.pyplot as plt
import numpy as np

def plotXY(x,y):
    #plt.plot(x,y,color='k',lw=10)
    plt.plot(x,y,color='k')

def plotOneLine(X,Y,len):
    x = np.linspace(X,X+len,10)
    y = np.zeros((x.shape[0],))+Y
    plotXY(x,y)

def StartDraw(N,startX,startY,len,inter):
    if N>0:
        sY = startY+inter
        plotOneLine(startX,sY,len/3)
        plotOneLine(startX+2*len/3,sY,len/3)
        print('plot s,t,sY = ',startX,len/3,sY,N)
        #print('plot s,t,sY = ',startX+2*len/3,len/3,sY,N)

        StartDraw(N-1,startX,sY,len/3,inter)
        StartDraw(N-1,startX+2*len/3,sY,len/3,inter)
    else:
        return

def main():
    recurse = 3  #iterated depths
    len = 30
    inter = 1
    startX,startY = 0,1
    plotOneLine(0,1,len)
    StartDraw(recurse,startX,startY,len,inter)

    #plt.axes().set_aspect('equal')
    plt.show()

if __name__ == "__main__":
    main()

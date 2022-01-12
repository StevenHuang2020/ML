#python3
#steven 04/03/2020
import matplotlib.pyplot as plt
import numpy as np

gAlpha = 100

def setPlot():
    plt.xticks(np.linspace(0,10,15))
    plt.yticks(np.linspace(-4000,4000,15))

def plotXY(x,y):
    plt.plot(x,y)
    #plt.show()

def getTreeBranch(startPt,slope):
    drawLen = 1
    x = np.linspace(startPt[0],startPt[0] + drawLen,10)
    b = startPt[1]-slope*startPt[0]
    y = slope*x + b
    ptLast = [x[-1],y[-1]]

    plotXY(x,y)
    #print('---------------------------start:,',startPt,'last:',ptLast)
    return ptLast

def tree(N, startPt, slope, level=1):
    level += 1
    if N > 0:
        lSlope = slope+gAlpha #slope*(1+gAlpha)
        rSlope = slope-gAlpha #slope*(1-gAlpha)
        #print('gLevel=',gLevel,'slope=',slope,'lSlope=',lSlope,'rSlope=',rSlope,startPt)

        ptLeft = getTreeBranch(startPt, lSlope)
        tree(N-1,ptLeft,lSlope)

        ptRight = getTreeBranch(startPt, rSlope)
        tree(N-1,ptRight,rSlope,level)
    else:
        return

def main():
    startPt = [0,0]
    slope = 0
    N=8
    #print(sys.getrecursionlimit())
    pt = getTreeBranch(startPt,slope)
    tree(N,pt,slope)
    setPlot()
    plt.show()

if __name__ == "__main__":
    main()

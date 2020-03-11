#python3
#steven 04/03/2020
import matplotlib.pyplot as plt
import numpy as np 

gAlpha = 200

def setPlot():
    plt.xticks(np.linspace(0,5,10))
    plt.yticks(np.linspace(-2300,2300,10))

def plotXY(x,y):
    plt.plot(x,y,color='g')
    #plt.show()

def sigmodCoefficient(x):
    return 1/(1+np.exp(-1*x)) #sigmod fuc

def getTreeBranchN(N,startPt,slope):
    drawLen = 1*sigmodCoefficient(N)

    x = np.linspace(startPt[0],startPt[0] + drawLen,10)
    b = startPt[1]-slope*startPt[0]
    y = slope*x + b
    ptLast = [x[-1],y[-1]]

    plotXY(x,y)
    return ptLast

def getTreeBranch(startPt,slope,left=True):
    drawLen = 1
    x = np.linspace(startPt[0],startPt[0] + drawLen,10)
    b = startPt[1]-slope*startPt[0]
    y = slope*x + b
    ptLast = [x[-1],y[-1]]

    plotXY(x,y)
    #print('---------------------------start:,',startPt,'last:',ptLast)
    return ptLast

def tree(N,startPt,slope):
    global gAlpha

    if N > 0:
        lSlope = slope+gAlpha #slope*(1+gAlpha)
        rSlope = slope-gAlpha #slope*(1-gAlpha)
        #print('slope=',slope,'lSlope=',lSlope,'rSlope=',rSlope,startPt)
        
        ptLeft = getTreeBranchN(N-1,startPt, lSlope)
        tree(N-1,ptLeft,lSlope)

        ptRight = getTreeBranchN(N-1,startPt, rSlope)
        tree(N-1,ptRight,rSlope)
    else:
        return

def main():
    startPt = [0,0]
    slope = 0
    N=5
    pt = getTreeBranchN(N,startPt,slope)
    tree(N,pt,slope)
    setPlot()
    plt.show()

if __name__ == "__main__":
    main()
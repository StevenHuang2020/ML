#python3
#steven 05/04/2020 Sierpiński triangle
#random point to simulate Sierpiński triangle
import matplotlib.pyplot as plt
import numpy as np 
import math

from plotTriangle4 import drawPolygon,getRatioPoint

def plotXY(x,y):
    plt.plot(x,y,color='k')
    #plt.plot(x,y)
    
def plotPointXY(x,y):
    plt.scatter(x,y,marker='.',color='k')
    #plt.plot(x,y)
    
def triangleStart(iter):    
    pt0 = np.array([0,0],dtype=np.float64)
    startLen=5
    pt1 = np.array([startLen/2,startLen/2*math.tan(math.pi/3)],dtype=np.float64)
    pt2 = np.array([startLen,0],dtype=np.float64)
    
    pts =[]
    pts.append(pt0)
    pts.append(pt1)
    pts.append(pt2)
    drawPolygon(pts)
    
    pt = np.array([2,1],dtype=np.float64)
    ptRandom = np.random.randint(3, size=(iter, ))
    for i in range(iter):
        plotPointXY(pt[0],pt[1])
        pt = getRatioPoint(pt,pts[ptRandom[i]])
    
    plt.axes().set_aspect('equal')
    plt.show()
    
def main():
    iter = 1000
    triangleStart(iter)
    
if __name__ == "__main__":
    main()
#python3
#steven 05/04/2020 Sierpiński triangle
#random point to simulate Sierpiński triangle
import matplotlib.pyplot as plt
import numpy as np
import math

from plotTriangle4 import drawPolygon,getRatioPoint
from plotTriangle4 import getSequenceCirclePoints,DrawTriangleLineByPt

def plotXY(x,y):
    plt.plot(x,y,color='k')
    #plt.plot(x,y)

def plotPointXY(x,y,ax,label=''):
    ax.scatter(x,y,marker='.',color='k')
    #ax.set_title(label)
    #plt.plot(x,y)

def triangleStart(iter=1000):
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
    ptRandom = np.random.randint(len(pts), size=(iter, ))

    ratios = [0.2,0.28,0.35,0.42,0.5,0.62]
    plt.suptitle("triangle fractal diff ratios")
    for id,rt in enumerate(ratios):
        ax = plt.subplot(3, 2, id+1)
        #ax.set_aspect('equal')
        ax.set_title('ratio='+str(rt))
        for i in range(iter):
            plotPointXY(pt[0],pt[1],ax)
            pt = getRatioPoint(pt,pts[ptRandom[i]],ratio=rt)

    #plt.axes().set_aspect('equal')
    plt.show()

def ploygonDrawLine(iter=100):
    numberPts = [5,6,8,9]
    for k,numberPt in enumerate(numberPts):
        ax = plt.subplot(2, 2, k+1)
        #ax.set_aspect('equal')
        pts = getSequenceCirclePoints(Num=numberPt,offset=2) #get start point list
        drawPolygon(pts)

        ptRandom = np.random.randint(1,numberPt-2, size=(iter, ))
        for i in range(iter):
            ptId1 = i%len(pts)
            ptId_next = (i+ptRandom[i]+1)%len(pts)
            DrawTriangleLineByPt(pts[ptId1],pts[ptId_next],ax)

    plt.show()
    pass

def ploygonDrawLine2(iter=100):
    numberPts = [16] #,6,8,9
    for k,numberPt in enumerate(numberPts):
        ax = plt.subplot(1, 1, k+1)
        ax.set_aspect('equal')
        pts = getSequenceCirclePoints(r=500,Num=numberPt,offset=2) #get start point list
        drawPolygon(pts)

        ptRandom = np.random.randint(1,numberPt-2, size=(iter, ))
        k=0
        for i in range(iter):
            #ptId1 = k
            ptId1 = i%len(pts)
            ptId_next = (k+ptRandom[i]+1)%len(pts)
            DrawTriangleLineByPt(pts[ptId1],pts[ptId_next],'k',ax)

            k=ptId_next

    plt.show()
    pass

def main():
    #triangleStart()
    #ploygonDrawLine()
    ploygonDrawLine2()

if __name__ == "__main__":
    main()

#python3
#steven 05/04/2020 Sierpiński triangle
#random start random points polygon
#ratio of getRatioPoint() indicate the division of line
import matplotlib.pyplot as plt
import numpy as np
import math

def plotXY(x,y,color='k',ax=None):
    c=color
    if ax:
        ax.plot(x,y,color=c)
    else:
        plt.plot(x,y,color=c)
        #plt.plot(x,y)

def DrawTriangleLineByPt(startPt,stopPt,color='k',ax=None):
    if startPt[0]>stopPt[0]: #switch
        startPt = startPt + stopPt
        stopPt = startPt - stopPt
        startPt = startPt -stopPt

    #print('s,t=',startPt,stopPt)
    x = np.linspace(startPt[0],stopPt[0],30)
    slope = (stopPt[1]-startPt[1])/(stopPt[0]-startPt[0])
    b = startPt[1]-slope*startPt[0]
    y = slope*x + b
    plotXY(x,y,color,ax)

def drawPolygon(points): #point sequence
    for i in range(1,len(points)):
        DrawTriangleLineByPt(points[i-1],points[i])
    DrawTriangleLineByPt(points[len(points)-1],points[0])

def trianglePolygon(points, N):
    if N>0:
        #draw big Polygon
        drawPolygon(points)

        #draw inner Polygon
        NPoints=[]
        for i in range(1,len(points)):
            NPoints.append(getRatioPoint(points[i-1],points[i]))
        NPoints.append(getRatioPoint(points[len(points)-1],points[0]))

        drawPolygon(NPoints)

        #recurse splited Polygon
        for i in range(1,len(points)):
            pts =[]
            pts.append(NPoints[i])
            pts.append(points[i])
            pts.append(NPoints[i-1])
            trianglePolygon(pts,N-1)

        pts =[]
        pts.append(NPoints[0])
        pts.append(points[0])
        pts.append(NPoints[len(NPoints)-1])
        trianglePolygon(pts,N-1)
    else:
        return

def getRatioPoint(pt1,pt2,ratio=0.35):
    #get point on the line of pt1 and pt2 acoording the ratio
    #when ratio=0.5, return the middle point
    #return np.mean( np.array([ pt1, pt2 ]), axis=0 )
    return pt1*ratio + pt2*(1-ratio)

def getRandomPoint(min=0, max = 5):
    return np.random.random((2,))*(max-min) + min  #[0,5)

def getRandom(min=0, max = 5):
    return np.random.random()*(max-min) + min

def circle(x,r=1):
    return np.sqrt(r**2-x**2)

def getRandomCirclePoint(r=1,positive=True):
    #get random point on circle,gurantee the polygon generated by these points is convex
    pt = np.array([0,0],dtype=np.float64)
    pt[0] = getRandom(min = -1*r, max = r)
    if positive:
        pt[1] = circle(pt[0], r=r)
    else:
        pt[1] = -1*circle(pt[0], r=r)
    return pt

def getSequenceCirclePoints(r=1,Num=5,offset=0):
    pts = []
    #offset = math.pi/(Num+1)
    for i in range(Num):
        pt = np.array([0,0],dtype=np.float64)
        angle = (i+1)*math.pi*2/Num + offset
        pt[0] = r*math.cos(angle)
        pt[1] = r*math.sin(angle)
        pts.append(pt)
    return pts

def triangleStart(N=3):
    pts = getSequenceCirclePoints(Num=5) #get start point list
    trianglePolygon(pts,N)

def main():
    recurse = 4  #iterated depths
    triangleStart(recurse)
    plt.axes().set_aspect('equal')
    plt.show()

if __name__ == "__main__":
    main()

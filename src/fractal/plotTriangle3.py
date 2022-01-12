#python3
#steven 05/04/2020 Sierpiński triangle
#random start three points traingle
#ratio of getRatioPoint() indicate the division of line
import matplotlib.pyplot as plt
import numpy as np

def plotXY(x,y):
    plt.plot(x,y,color='k')
    #plt.plot(x,y)

def DrawTriangleLineByPt(startPt,stopPt):
    if startPt[0]>stopPt[0]: #switch
        startPt = startPt + stopPt
        stopPt = startPt - stopPt
        startPt = startPt -stopPt

    #print('s,t=',startPt,stopPt)
    x = np.linspace(startPt[0],stopPt[0],30)
    slope = (stopPt[1]-startPt[1])/(stopPt[0]-startPt[0])
    b = startPt[1]-slope*startPt[0]
    y = slope*x + b
    plotXY(x,y)

def triangle(pt0, pt1,pt2, N):
    if N>0:
        #draw big triangle
        DrawTriangleLineByPt(pt0,pt1)
        DrawTriangleLineByPt(pt1,pt2)
        DrawTriangleLineByPt(pt2,pt0)

        #draw inner triangle
        N_pt0 = getRatioPoint(pt0,pt1)
        N_pt1 = getRatioPoint(pt1,pt2)
        N_pt2 = getRatioPoint(pt2,pt0)
        #print('N_pt0, N_pt1, N_pt2 = ', N_pt0, N_pt1, N_pt2)
        DrawTriangleLineByPt(N_pt0,N_pt1)
        DrawTriangleLineByPt(N_pt1,N_pt2)
        DrawTriangleLineByPt(N_pt2,N_pt0)

        #recurse three splited triangle
        triangle(pt0,N_pt0,N_pt2,N-1)
        triangle(N_pt0,pt1,N_pt1,N-1)
        triangle(N_pt2,N_pt1,pt2,N-1)
    else:
        return

def getRatioPoint(pt1,pt2,ratio=0.5):
    #get point on the line of pt1 and pt2 acoording the ratio
    #when ratio=0.5, return the middle point
    #return np.mean( np.array([ pt1, pt2 ]), axis=0 )
    return pt1*ratio + pt2*(1-ratio)

def getRandomPoint(min=0, max = 5):
    return np.random.random((2,))*(max-min) + min  #[0,5)

def triangleStart(N=3):
    pt0 = getRandomPoint()
    pt1 = getRandomPoint()
    pt2 = getRandomPoint()
    #print('pt0,pt1,pt2 = ',pt0,pt1,pt2)
    triangle(pt0,pt1,pt2,N)   #draw triangle

def main():
    recurse = 5  #iterated depths
    triangleStart(recurse)
    plt.axes().set_aspect('equal')
    plt.show()

if __name__ == "__main__":
    main()

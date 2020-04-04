#python3
#steven 04/04/2020
import matplotlib.pyplot as plt
import numpy as np 
import math

gSlop1 = math.tanh(math.pi/3)
slopeUp= [math.tanh(math.pi/3),-1*math.tanh(math.pi/3),0]
slopeDown= [0, math.tanh(math.pi/3), -1*math.tanh(math.pi/3)]

def plotXY(x,y):
    #plt.plot(x,y,color='g')
    plt.plot(x,y)
    
def DrawTriangleLineByPt(startPt,stopPt,slope):
    if startPt[0]>stopPt[0]: #switch
        startPt = startPt + stopPt
        stopPt = startPt - stopPt
        startPt = startPt -stopPt
        
    x = np.linspace(startPt[0],stopPt[0],10)
    b = startPt[1]-slope*startPt[0]
    y = slope*x + b
    plotXY(x,y)

def triangle(pt0, pt1,pt2,slopes):
    DrawTriangleLineByPt(pt0,pt1,slopes[0])
    DrawTriangleLineByPt(pt1,pt2,slopes[1])
    DrawTriangleLineByPt(pt2,pt0,slopes[2])

def triangleStart(startPt,lineLen,N):
    if N>0:
        pt0,pt1,pt2 = np.array([0,0],dtype=np.float64),\
                        np.array([0,0],dtype=np.float64),\
                        np.array([0,0],dtype=np.float64)
        
        pt0 = startPt
        pt1[0] = pt0[0]+lineLen/2
        pt1[1] = pt0[1]+lineLen/2*gSlop1
        pt2[0] = pt0[0] + lineLen
        pt2[1] = pt0[1]
        #print(pt0,pt1,pt2)
        
        triangle(pt0,pt1,pt2,slopeUp)
        
        N_pt0 = pt0.copy()
        N_pt0[0] += lineLen/4
        N_pt0[1] += (lineLen/4*gSlop1)
        
        N_pt1 = N_pt0.copy()
        N_pt1[0] += lineLen/2
        
        N_pt2 = pt0.copy()
        N_pt2[0] += lineLen/2
        
        #print(N_pt0, N_pt1, N_pt2)
        triangle(N_pt0, N_pt1, N_pt2,slopeDown)
        
        triangleStart(pt0,lineLen/2,N-1)
        triangleStart(N_pt0,lineLen/2,N-1)
        triangleStart(N_pt2,lineLen/2,N-1)
    else:
        return
   
def main():
    startPt = np.array([0,0],dtype=np.float64)
    startLen=5
    recurse = 8
    triangleStart(startPt, startLen,recurse)
    plt.show()
    
if __name__ == "__main__":
    main()
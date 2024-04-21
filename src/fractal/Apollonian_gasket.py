#python3
#steven 24/04/2020 Apollonian gasket
#Reference:https://en.wikipedia.org/wiki/Apollonian_gasket
import numpy as np
import matplotlib.pyplot as plt

def getCircle(pt,r):
    x = np.linspace(pt[0] - r, pt[0] + r, 50)
    y = np.linspace(pt[1] - r, pt[1] + r, 50)
    X, Y = np.meshgrid(x, y)
    return X, Y, (X-pt[0])**2 + (Y-pt[1])**2 - r**2

# def plotCircle(ax):
#     x = np.linspace(-1.0, 1.0, 100)
#     y = np.linspace(-1.0, 1.0, 100)
#     X, Y = np.meshgrid(x, y)
#     F = X ** 2 + Y ** 2 - 1
#     ax.contour(X, Y, F, [0],colors=['red'])
#     ax.set_aspect(1)

def plotCircle(ax, param):
    X,Y,F = param
    ax.contour(X, Y, F, [0])
    ax.set_aspect(1)

def plotCirclePts(ax,params):
    for i in params:
        pt = i[0]
        r = i[1]
        plotCircle(ax, getCircle((pt[0],pt[1]),r))

def plotXY(x,y):
    plt.plot(x,y)
    #plt.show()

def plotLine(pt1,pt2):
    slope = (pt2[1]-pt1[1])/(pt2[0]-pt1[0])
    b = pt1[1] - slope*pt1[0]
    x = np.linspace(pt1[0],pt2[0],20)
    y = slope*x + b
    plotXY(x,y)

def draw_1():
    ax = plt.subplot(1,1,1)

    x1,y1,r1 = -1, 0, 2
    x2,y2,r2 = 3, 2, 2
    x3,y3,r3 = 2, -2, 1

    pts = []
    #pts.append(((-1,0),1))
    #pts.append(((1, 0),2))
    #pts.append(((3, 0),np.sqrt(4**2)-1))
    pts.append(((x1,y1),r1))
    #pts.append(((x2,y2),r2))
    rTengen = np.sqrt((x1-x2)**2+(y1-y2)**2)-r1
    pts.append(((x2,y2),rTengen))

    rTengen2 = np.sqrt((x1-x3)**2+(y1-y3)**2)-r1
    pts.append(((x3,y3),rTengen2))

    plotCirclePts(ax,pts)

    plotLine((x1,y1),(x2,y2))
    plotLine((x2,y2),(x3,y3))
    plotLine((x3,y3),(x1,y1))
    plt.show()

def draw_2():
    ax = plt.subplot(1,1,1)
    pts = []
    pts.append(((-1,0),1))
    pts.append(((1, 0),1))
    pts.append(((0, np.sqrt(3)),1))

    plotCirclePts(ax,pts)

    x = np.tan(np.pi/6)
    r = 1/np.cos(np.pi/6)
    plotCircle(ax, getCircle((0, x), r - 1))
    plotCircle(ax, getCircle((0, x), r + 1))
    plt.show()

def main():
    #draw_1()
    draw_2()

if __name__=='__main__':
    main()

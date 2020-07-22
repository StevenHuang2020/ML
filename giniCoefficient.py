#Python3 Steven
#simualtion distrubutions for people's income 
#https://en.wikipedia.org/wiki/Gini_coefficient
import numpy as np
import matplotlib.pyplot as plt
from genDistributions import *
from distributions import *
from plotCommon import *

def calLineSlopAndBias(startPt,stopPt):
    slop = (stopPt[1]-startPt[1])/(stopPt[0]-startPt[0])
    b = stopPt[1]-slop*stopPt[0]
    return slop,b

def calDistribution(dis):
    dis = np.abs(dis)
    disOrdered = sorted(dis)
    #plot(range(len(dis)), dis)
    cumDis = np.cumsum(disOrdered)
    
    l = len(cumDis)
    slop,b = calLineSlopAndBias([0,cumDis[0]],[len(cumDis)-1,cumDis[len(cumDis)-1]])
    equalDis = [ slop*i+b for i in range(l)]
    
    gini = (np.sum(equalDis)-np.sum(cumDis))/np.sum(equalDis)
    return dis,disOrdered,cumDis,equalDis,gini

def plotAll(dis,name=''):
    dis,disOrdered,cumDis,equalDis,gini = calDistribution(dis)
    
    fmt = '{:24} | {:18}'
    print( fmt.format('Distribution:'+name,'Gini coefficient:' + str(round(gini,4))))
    
    plotIncomeDistribution(name,dis,disOrdered)
    # plotLorenzCurve(cumDis,equalDis,gini)
    plotWhole(name,dis,disOrdered,cumDis,equalDis,gini)
   
gSaveBasePath=r'.\images\\' 
def plotWhole(name,dis,disOrdered,cumDis,equalDis,gini):
    plt.figure(figsize=(9,4.2))
    
    ax = plt.subplot(1, 2, 1)
    ax.plot(dis,label='Income ' + name + ' distribution')
    ax.plot(disOrdered,label='Income ordered')    
    ax.set_title("Income")
    ax.legend()
    
    ax = plt.subplot(1, 2, 2)
    z0 = np.zeros(len(cumDis))
    x = range(len(cumDis))
    c1 = tuple(ti/255 for ti in (176,196,214))   
    c2 = tuple(ti/255 for ti in (128,179,255)) 
    ax.fill_between(x,equalDis,cumDis, color=c1)# where=x<=1, facecolor='green'
    ax.fill_between(x,z0,cumDis, color=c2)
    ax.plot(cumDis,label='Cumlative Income')
    ax.plot(equalDis,linestyle='dashed',label='Equal Income')
    ax.set_title('Income Lorenz Curve,Gini='+str(round(gini,4)))
    ax.legend()
    
    plt.subplots_adjust(left=0.05, bottom=0.07, right=0.97, top=0.90, wspace=None, hspace=None)
    plt.savefig(gSaveBasePath + name + '_simulationGini' + '.png')
    plt.show()

def plotIncomeDistribution(name,dis,disOrdered):
    ax = plt.subplot(1, 1, 1)
    #ax.plot(dis,label='Income distribution')
    #ax.plot(disOrdered,label='Income ordered')    
    ax.hist(dis,bins=60, rwidth=0.72, label='Income',alpha=0.75)
    #plt.hist(x, 50, density=True, facecolor='g', alpha=0.75)
    
    ax.set_title("Income " + name + ' Distribution')
    ax.legend()
    plt.savefig(gSaveBasePath + name + '_simulationIncome' + '.png')
    plt.show()
    
def plotLorenzCurve(cumDis,equalDis,gini): #https://en.wikipedia.org/wiki/Lorenz_curve    
    ax = plt.subplot(1, 1, 1)
    
    z0 = np.zeros(len(cumDis))
    x = range(len(cumDis))
    c1 = tuple(ti/255 for ti in (176,196,214))   
    c2 = tuple(ti/255 for ti in (128,179,255)) 
    plt.fill_between(x,equalDis,cumDis, color=c1)# where=x<=1, facecolor='green'
    plt.fill_between(x,z0,cumDis, color=c2)
    ax.plot(cumDis,label='Cumlative Income')
    ax.plot(equalDis,linestyle='dashed',label='Equal Income')
    ax.set_title('Income Lorenz Curve,Gini='+str(round(gini,4)))
    ax.legend()
    plt.show()

    
def main():
    N=500
    start=0
    width=1
    dis = genUniform_distribution(N,start,width)
    #dis = Uniform_distribution(range(N),0,1)
    #print(dis,'sum=',np.sum(dis))
    #plot(np.linspace(0,1,N), dis)
    #plot(range(N), dis)
    plotAll(dis,name='Uniform')

    dis = genNormal_distribution(N,1)
    #print(dis,'sum=',np.sum(dis))
    plotAll(dis,name='Normal')
    
    dis = genGamma_distribution(N)
    plotAll(dis,name='Gamma')
    
    dis = genExp_distribution(N)
    plotAll(dis,name='Exponential')
    
    dis = genPoisson_distribution(N)
    plotAll(dis,name='Poisson')
    
    dis = genBinomial_distribution(N)
    plotAll(dis,name='Binomial')
    
    dis = genBernoulli_distribution(N)
    plotAll(dis,name='Bernoulli')
    
    
if __name__=='__main__':
    main()
    
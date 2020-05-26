#python3
#steven 01/03/2020
#coin distribution testing
import numpy as np 
import matplotlib.pyplot as plt

def randomSeries(N): #generate series number between 0~1
    return np.random.rand(N)

def plotNormalDistribution():
    x = np.linspace(-1,1,50)
    y = 1/np.sqrt(2*np.pi) * np.exp(-0.5*x**2)
    plotDistribution(x,y,'Normal Distribution')
    
def plotDistribution(x,y,name):
    plt.title(name)
    plt.plot(x,y)
    plt.xlabel('Proability of event for every batch')
    plt.ylabel('Frequency of Proability')
    plt.show()

def calculateDistribution(distributions,name):
    #print(distributions)
    distributions_set = list(set(distributions)) #remove repeat prob 
    distributions_set.sort()
    #print(distributions_set)

    distributions_setCount=[]  #calculate repeat times for erery prob
    for i in distributions_set:
        distributions_setCount.append(distributions.count(i))
    #print(distributions_setCount)

    distributions_setProb = []  #calculate frequency prob
    sum = np.sum(distributions_setCount)
    for i in distributions_setCount:
        distributions_setProb.append(i/sum)
    #print(distributions_setProb)

    maxFrequency = np.max(distributions_setProb)
    index = distributions_setProb.index(maxFrequency)
    print('index=',index,'prob=',distributions_set[index].round(4),'max frequency:',maxFrequency.round(4))

    plotDistribution(distributions_set,distributions_setProb,name)


def main():
    #plotNormalDistribution()
    #experment start
    T = 300000  #times of batch
    N = 200     #coin test times every batch
    
    distributions = []
    for _ in range(T):
        x = randomSeries(N)
        res = np.where(x > 0.5, 1, 0) #x>0.5 stand for upper side of coin throwing
        prob = np.sum(res)/len(res)
        distributions.append(prob)
    
    return calculateDistribution(distributions,'Coins Distribution')

if __name__ == "__main__":
    main()

#steven 01/03/2020
#dice distribution testing
import numpy as np 
from coinDistrbution import calculateDistribution

def randomSeriesInt(N):
    """generate series dice number """
    return np.random.randint(1,7,N)

def main():
    #experment start

    #test 1
    """
    N = 20000   #coin test times erery batch
    distributions = list(randomSeriesInt(N))
    return calculateDistribution(distributions)
    """

    #test2
    T = 50000  #times of batch
    N = 100    #coin test times erery batch
    
    distributions = []
    for _ in range(T):
        x = randomSeriesInt(N)
        res = np.where(x == 1, 1, 0) #stand for number 1 of dice throwing
        prob = np.sum(res)*1.0/len(res)
        distributions.append(prob)
        #print(x)

    #print(distributions)
    return calculateDistribution(distributions,'Dice Distribution')

if __name__ == "__main__":
    main()

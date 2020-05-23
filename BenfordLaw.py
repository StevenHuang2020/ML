#Steven 05/05/2020 Benford's Law
#reference: https://en.wikipedia.org/wiki/Benford%27s_law
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def Benford(N):
    return np.log10(1+1/N).round(4)

def numFirstStr(value):
    return str(value)[0]

def rangeIntN(N=100):
    return np.random.randint(100, size=N)
    
def softMaxFuc(X):
    return X/np.sum(X)

def getTestData():
    f = r'.\db\coronavirous_2020-05-23.csv'
    df = pd.read_csv(f)
    df.set_index(["Location"], inplace=True)
    #print(df.head())
    #data = df['Confirmed']
    data = df.iloc[:,[1]]
    return data.values

def plotProb(benford,prob):
    x=np.arange(len(prob))+1
    print(x)
    plt.plot(x,benford,label='Benford')
    plt.plot(x,prob,label='Actual')
    plt.bar(x,height=prob,label='Actual')
    plt.scatter(x,benford,color='k',label='Benford')
    
    plt.legend()
    plt.show()
    
def main():
    benford = list(map(Benford, [i+1 for i in range(9)]))
    #benford = Benford(np.arange(1,10))
    #print(benford)
    
    prob = np.zeros((9,))
    series = getTestData()   #rangeIntN(N=10)
    series = series.flatten()
    print(series.shape)
    for i in range(len(series)):
        #print(series[i],numFirstStr(series[i]))
        id = int(numFirstStr(series[i]))
        if id >0:
            prob[id-1] += 1
    print(prob)
    actBen = softMaxFuc(prob).round(4)
    
    print('benford=', benford)
    print('actBen=', actBen)
    plotProb(benford,actBen)
    
if __name__=='__main__':
    main()
    
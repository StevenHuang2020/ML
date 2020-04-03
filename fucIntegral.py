#steven 02/03/2020
#function integral to calculate area of curve and x axis
import numpy as np 
import matplotlib.pyplot as plt

def softmaxFuc(x):
    #x = [2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
    softmax = np.exp(x)/np.sum(np.exp(x))
    print(softmax)
    print(np.sum(softmax))
    return softmax

def plot(x,y):
    plt.plot(x,y)
    plt.show()

def fun(x):
    #return x 
    #return x**2
    #return np.sqrt(1-x**2)   #y=sqrt(1-x**2)  #circle equations calculate pi
    #return np.exp(x)
    #return 1/(1 + np.power(np.e,-1*x))
    return np.power(x,x)

def IntegralFuc():
    """calculate area y=x^2  form x = 0 to 1
    divide 0~1 to N shares,  erevy part considered as a rectangle.
    """
    N = 100000
    s = 0
    for i in range(N):
        x = 1/N
        y = fun(i/N)
        s += x*y
    
    print(s)
    #pi = s*4 #when fuc = np.sqrt(1-x**2) 
    #print(pi)
    return s
	
def fillColor():
    x = np.linspace(0,1.5, 100)
    
    y1 = np.zeros(len(x))
    y2 = fun(x)
    #new_ticks = np.linspace(0,2,5) 
    #print(new_ticks)
    #plt.xticks(new_ticks)

    plt.plot(x,y1,c='b',alpha=0.5)
    plt.plot(x,y2,c='b',alpha=0.5)
    
    plt.fill_between(x,y1,y2,where=x<=1,facecolor='green')
    #plt.grid(True)
    plt.show()

def main():
    s = IntegralFuc()
    fillColor()

    #x = np.linspace(-5,5, 50)
    #y = fun(x)
    #y = softmaxFuc(x)
    #plot(x,y)

if __name__ == '__main__':
    main()

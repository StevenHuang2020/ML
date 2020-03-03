#steven 02/03/2020
#function integral to calculate area of curve and x axis
import numpy as np 
import matplotlib.pyplot as plt

def fun(x):
    #return x 
    return x**2

def IntegralFuc():
    """calculate area y=x^2  form x = 0 to 1
    divide 0~1 to N shares,  erevy part considered as a rectangle.
    """
    N = 10000

    s = 0
    for i in range(N):
        x = 1/N
        y = fun(i/N)
        s += x*y
    
    print(s)
	
def fillColor():
    x = np.linspace(0,1.5, 100)
    
    y1 = np.zeros(len(x))
    y2 = fun(x)
    
    #new_ticks = np.linspace(0,2,5) 
    #print(new_ticks)
    #plt.xticks(new_ticks)

    plt.plot(x,y1,c='b',alpha=0.5)
    plt.plot(x,y2,c='b',alpha=0.5)
    

    plt.fill_between(x,y1,y2,where=x<1,facecolor='green')
    #plt.grid(True)
    plt.show()


def main():
   IntegralFuc()
   fillColor()

if __name__ == '__main__':
    main()

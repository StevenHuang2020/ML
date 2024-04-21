#Python3 Steven
import numpy as np
import matplotlib.pyplot as plt

def plotPowerFuc():
    x = np.linspace(-4,4,100)
    e = range(-1,4)
    for i in e:
        y = x**i
        plt.plot(x,y,label='y=x**%s'%i,linewidth=2)

    plt.xlim(x[0],x[-1])
    plt.ylim(-20,20)
    plt.legend()
    plt.show()

def plotLine(n=80):
    for k in range(n):
        #plt.plot([0, np.cos(2*np.pi*k/n)], [0, np.sin(2*np.pi*k/n)])
        plt.plot([0, np.cos(2*np.pi*k/n)], [0, np.sin(2*np.pi*k/n)],color=[k/n,k/n,k/n])
        #plt.plot([0,k*np.cos(2*np.pi*k/n)], [0, k*np.sin(2*np.pi*k/n)],color=[k/n,k/n,k/n])
        #plt.plot([0,k*np.cos(2*np.pi*k/n)], [0, k*np.sin(2*np.pi*k/n)])

    plt.axis('square')
    plt.show()

def complexRoot(n=2):
    #Z**n=1
    for k in range(n):
        yield np.exp(2*np.pi*1j*k/n)

def plotCompexRoot(n=50):
    for k in range(n):
        #z = np.exp(2*np.pi*1j*k/n)
        z = k*np.exp(2*np.pi*1j*k/n)
        #plt.plot([0,np.real(z)], [0,np.imag(z)])
        plt.plot([0,np.real(z)], [0,np.imag(z)],color=[0,0,0])

    plt.axis('square')
    plt.show()

def plotTrigonometry():
    t = np.linspace(0,8*np.pi,1000)
    r1 = np.random.rand()
    r2 = np.random.rand()

    x = np.cos(r1*t)
    y = np.cos(r2*t)
    plt.plot(x,y,'k')
    plt.title('r1=%s,r2=%s'%(np.round(r1,2),np.round(r2,2)))
    plt.axis('square')
    plt.show()

def derivative(f,x,h=0.0001): #slop
    return (f(x+h)-f(x))/h

def func(x):
    return x**8
    #return x**3

def plotTangent():
    x = np.linspace(-1,1,50)
    xT = np.linspace(-1,1,100)
    bound=[-2,2]
    for i in xT:
        y=func(i)
        slope = derivative(func,i)
        b = y-slope*i
        plt.plot([bound[0],bound[1]],[slope*bound[0]+b,slope*bound[1]+b],color=[abs(i)/3,abs(i)/2,abs(i)/3])
    #plt.axis('square')
    plt.axis('off')
    plt.plot(x,func(x))
    plt.xlim(x[0],x[-1])
    plt.ylim(-3,3)
    plt.show()

def main():
    #plotPowerFuc()
    #plotLine()
    #plotCompexRoot()
    #plotTrigonometry()
    plotTangent()

if __name__=='__main__':
    main()

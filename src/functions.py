import numpy as np
import math
from scipy.special import gamma,factorial,beta
from scipy import integrate
from distributions import *


def derivative(f,x,h=0.0001): #one parameter fuction derivative
    return (f(x+h)-f(x))/h

def derivative2(f,x,h=0.0001): #Second Derivative
    return (derivative(f,x+h,h) - derivative(f,x,h))/h

def normalDistribution(x):
    return 1/np.sqrt(2*np.pi) * np.exp(-0.5*x**2)

def softmaxFuc(x):
    softmax = np.exp(x)/np.sum(np.exp(x))
    #print(softmax)
    #print(np.sum(softmax))
    return softmax

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def log(x):
    return np.log(x)

def entropy(x):
    return -1*x*np.log(x)

def exp(x):
    return np.exp(x)

def circle(x):
    return np.sqrt(1-x**2)*randomSymbol(len(x))

def heart(x): #heart equation: x**2+ (5*y/4 - sqrt(abs(x)))**2 = 1
    a = np.sqrt(1-x**2)*randomSymbol(len(x)) + np.sqrt(abs(x))
    return a*4/5

def sawtoothWave(x): # y = t-floor(t) or y = t%1
    #return x%1
    #return x%1 + x
    #return x%1 + np.sin(x)*x
    a = 10 #perid
    return 2*(x/a - np.floor(0.5+x/a))

def powerX(x):
    #return np.power(x,x)
    return powerAX(x,x)

def powerAX(a,x):
    return np.power(a,x)

def logisticMap2(r=1.5,x0=0.8,N=10): #x:= r*x(1-x)
    maps=[]
    a = x0
    while N>0:
        #print(a,' ',end='')
        maps.append(a)
        a = r*a*(1-a)**3
        N -= 1

    return maps

def logisticMap(r=1.5,x0=0.8,N=10): #x:= r*x(1-x)
    maps=[]
    a = x0
    while N>0:
        #print(a,' ',end='')
        maps.append(a)
        a = r*a*(1-a)
        N -= 1

    return maps

def randomSymbol(length): #-1,1,1,-1,1,-1...
    f = np.zeros(length)+np.where(np.random.random(length)>0.5,1.0,-1.0)
    return f

def Divisorfunction(param): #https://en.wikipedia.org/wiki/Divisor_function#Definition
    def getDivsorList(N):
        for i in range(1,N+1):
            #print('i=',i)
            if N%i == 0:
                yield i

    N = param[0]
    p = param[1]
    def powerF(x):
        #print('x=',x,np.power(x,p))
        return np.power(x,p)

    l = list(map(powerF, list(getDivsorList(N))))
    return np.sum(l)


def EulerTotients(N=10):#https://en.wikipedia.org/wiki/Euler%27s_totient_function
    def gcd(p,q): # Create the gcd of two positive integers.
        while q != 0:
            p, q = q, p%q
        return p

    def is_coprime(x, y):
        return gcd(x, y) == 1

    def Totients(x):
        if x == 1:
            return 1
        n = [y for y in range(1,x) if is_coprime(x,y)]
        #print('n=',n)
        return len(n)

    return Totients(N)

def PrimeNumbers(N=10): #https://en.wikipedia.org/wiki/Prime-counting_function
    def isPrime(num):
        if num>1:
            for n in range(2, num):
                if (num % n) == 0:
                    return False
            return True
        return False

    def getPrime():
        for i in range(2,N+1):
            if isPrime(i):
                yield i

    l = list(getPrime())
    print('l=', len(l), l)
    return len(l)

def Mobiusfunction(N=10): #https://en.wikipedia.org/wiki/M%C3%B6bius_function
    def IsSquareFreeIntegers(x):
        for n in range(2, int(np.sqrt(x)+1)):
            if (x % n**2) == 0:
                return False
        return True

    def SquareFreeIntegers2():
        squareFreeNumbers=[]
        for val in range(1, N + 1):
            if IsSquareFreeIntegers(val):
                squareFreeNumbers.append(val)
        return squareFreeNumbers

    def SquareFreeIntegers():
        bSqure=False
        for val in range(1, N + 1):
            for n in range(2, int(np.sqrt(N)+1)):
                bSqure=False
                if (val % n**2) == 0:
                    bSqure=True
                    break
            if not bSqure:
                yield val

    def Mobius(x):
        if IsSquareFreeIntegers(x):
            nPrime = PrimeNumbers(x)
            print('x,nPrime=',x,nPrime)
            if nPrime % 2 ==0:
                return 1
            return -1
        return 0

    #squareFreeNumbers = SquareFreeIntegers2() #[i for i in SquareFreeIntegers()]
    #print('squareFreeNumbers=',len(squareFreeNumbers),squareFreeNumbers)
    #l = list(map(Mobius, [i for i in range(N)]))
    #print('l=',len(l),l)
    return Mobius(N)

def LegendreFunction(x,n=0): #https://en.wikipedia.org/wiki/Legendre_function
    if n == 0:
        return 0.5*np.log((1+x)/(1-x))
    elif n == 1:
        return x*LegendreFunction(x,0) -1 #P1(x) = x
    return ((2*n-1)/n)*x*LegendreFunction(x,n-1) - ((n-1)/n)*LegendreFunction(x,n-2)

def ScorersFunctionGi(x): #https://en.wikipedia.org/wiki/Scorer%27s_function
    def gi(t):
        return (1/np.pi)*np.sin((1/3)*t**3 + x*t)
    # return integrate.quad(gi, 0, 1)[0]
    def hi(t):
        return (1/np.pi)*np.exp((-1/3)*t**3 + x*t)

    # rGi = integrate.quad(gi, 0, np.inf,epsabs=1.49e-8,
    #                epsrel=1.49e-8, maxp1=50, limit=50)[0]
    # rHi = integrate.quad(hi, 0, np.inf,epsabs=1.49e-8,
    #                epsrel=1.49e-8, maxp1=50, limit=50)[0]
    # return rGi,rHi
    return integrate.quad(gi, 0, np.inf)[0], integrate.quad(hi, 0, np.inf)[0]

def Logarithmic_integral(x): #https://en.wikipedia.org/wiki/Logarithmic_integral_function
    def Li(t):
        return 1/np.log(t)

    return integrate.quad(Li, 0, x)[0]

def Exponential_integral(x): #https://en.wikipedia.org/wiki/Exponential_integral
    def exp_in(t):
        return -1*np.exp(-1*t)/t

    return integrate.quad(exp_in, -1*x, np.inf)[0]

def Trigonometric_integral(x): #https://en.wikipedia.org/wiki/Trigonometric_integral
    def Si(t):
        return np.sin(t)/t
    def Ci(t):
        return -1*np.cos(t)/t

    si = integrate.quad(Si, 0, x)[0]
    ci = integrate.quad(Ci, x, np.inf)[0]
    return si,ci

def KL_divergence(x):#https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
    p = NormalDistribution(x)
    q = NormalDistribution(x,u=1)

    def KL_integratePQ(t):
        return p*np.log(p/q)

    def KL_integrateQP(t):
        return q*np.log(q/p)

    return KL_integratePQ(x),KL_integrateQP(x)
    #return integrate.quad(KL_integrate, -100, x)[0]
    #return integrate.quad(KL_integrate, -1*np.inf, np.inf)[0]
    #return integrate.quad(KL_integrate, -3, 4)[0]

#python3 steven 03/04/2020
#All found activation fuction in ML
#Reference: https://en.wikipedia.org/wiki/Activation_function
import numpy as np

def Identity(x):
    return x

def sigmoid(x): #aka Logistic
    return 1 / (1 + np.exp(-x))

def Binary_step(x):
    return np.where(x<0,0,1)

def Tanh(x): 
    return np.tanh(x)
    
def SQNL(x): #https://ieeexplore.ieee.org/document/8489043
    y = np.zeros((len(x),))
    
    l = np.where(x > 2.0)
    if len(l) != 0:
        y[l[0]]=1
    l = np.where(x<=2.0)
    if len(l) != 0:
        y[l[0]]=x[l[0]]-x[l[0]]**2/4
    l = np.where(x<0)
    if len(l) != 0:
        y[l[0]]=x[l[0]]+x[l[0]]**2/4
    l = np.where(x<-2.0)
    if len(l) != 0:
        y[l[0]]=-1
        
    return y
    
def ArcTan(x):
    return np.arctan(x)
    
def ArcSinH(x):
    return np.arcsinh(x)

def Softsign(x): #http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
    return x/(1+abs(x))

def ISRu(x,alpha=1.0): #https://arxiv.org/pdf/1710.09967.pdf
    return x/np.sqrt(1+alpha*x**2)

def ISRLu(x,alpha=1.0):#https://arxiv.org/pdf/1710.09967.pdf
    y = np.zeros((len(x),))
    l = np.where(x < 0)
    if len(l) != 0:
        y[l[0]]=ISRu(x[l[0]],alpha)
        
    l = np.where(x >= 0)
    if len(l) != 0:
        y[l[0]]=x[l[0]]
        
    return y
    
def PLu(x,alpha=0.1,c=1): #https://arxiv.org/pdf/1809.09534.pdf
    c = np.zeros((x.shape))+c    
    return np.maximum(alpha*(x+c)-c, np.minimum(alpha*(x-c)+c, x))

def Relu(x): #https://www.cs.toronto.edu/~fritz/absps/reluICML.pdf
    return np.maximum(np.zeros((x.shape)),x)

def BReLu(x):#https://arxiv.org/pdf/1709.04054.pdf
    if len(x) % 2 ==0:
        return Relu(x)
    else:
        return -1*Relu(-1*x)
    
def LeakyRelu(x):#https://www.semanticscholar.org/paper/Rectifier-Nonlinearities-Improve-Neural-Network-Maas/367f2c63a6f6a10b3b64b8729d601e69337ee3cc
    return np.where(x<0,0.01*x,x)
    
def PRelu(x,alpha=.1): #https://arxiv.org/pdf/1502.01852.pdf
    return np.where(x<0,alpha*x,x)
    
def RRelu(x,alpha=.1): #https://arxiv.org/pdf/1505.00853.pdf
    return np.where(x<0,alpha*x,x)
    
def GELu(x):#https://arxiv.org/pdf/1606.08415.pdf   #https://github.com/hendrycks/GELUs
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))

def ELu(x,alpha=1.0): #https://arxiv.org/pdf/1511.07289.pdf
    return np.where(x<0, alpha*(np.exp(x)-1), x)
    
def SELu(x,alpha=1.67326,la=1.0507): #https://arxiv.org/pdf/1706.02515.pdf
    return la*ELu(x,alpha)

def softPlus(x): #http://proceedings.mlr.press/v15/glorot11a/glorot11a.pdf
    return np.log(1+np.exp(x))

def BentIdentity(x):
    return 0.5*(np.sqrt(x**2+1)-1)+x

def SoftExponential(x,alpha=1.0): #https://arxiv.org/pdf/1602.01321.pdf
    if alpha<0:
        return -1*(np.log(1-alpha*(x+alpha))/alpha)
    elif alpha == 0:
        return x
    else:
        return (np.exp(alpha*x)-1)/alpha + alpha

def SoftClipping(x,alpha=1.0): #https://arxiv.org/pdf/1810.11509.pdf
    return (1/alpha)*np.log((1+np.exp(alpha*x))/(1+np.exp(alpha*(x-1))))

def Squashing(x,alpha=0.5,lam=1.0,beta=1): #https://arxiv.org/pdf/1910.02486.pdf
    z1 = np.exp(beta*(x-(alpha-lam*0.5)))
    z2 = np.exp(beta*(x-(alpha+lam*0.5)))
    return (1/(lam*beta))*np.log((1+z1)/(1+z2))

def Sinusoid(x):
    return np.sin(x)

def Sinc(x):
    return np.where(x==0,1,np.sin(x)/x)

def Gaussian(x):
    return np.exp(-1*x**2)

def SQ_RBF(x):
    y = np.zeros((len(x),))
    l = np.where(abs(x) <= 1)
    if len(l) != 0:
        y[l[0]]=1-0.5*x[l[0]]**2
        
    l = np.where(abs(x) > 1)
    if len(l) != 0:
        y[l[0]]=0.5*(2-abs(x[l[0]]))**2
        
    l = np.where(abs(x) > 2)
    if len(l) != 0:
        y[l[0]]=0
        
    return y

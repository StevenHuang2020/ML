#python3 Steven 11/15/2020, Auckland,NZ
#Loss function
#Reference: https://en.wikipedia.org/wiki/Loss_function
import numpy as np

#https://papers.nips.cc/paper/2008/file/f5deaeeae1538fb6c45901d524ee2f98-Paper.pdf
def Least_squares(x): #LS
    return (1-x)**2

def Modified_LS(x):
    def loss(x):
        return (np.max(1-x, 0))**2
    return list(map(loss, [i for i in x]))

def SVM_Loss(x):
    def loss(x):
        return np.max(1-x, 0)
    return list(map(loss, [i for i in x]))

def Boosting_Loss(x):
    return np.exp(-1*x)

def LogisticRegression(x):
    return np.log(1+np.exp(-1*x))

def Savage_loss(x):
    return 1/(1+np.exp(2*x))**2

def zero_one(x):
    y = np.zeros_like(x)
    l = np.where(x < 0)
    if len(l) != 0:
        y[l[0]]=1
    return y

#python3 Steven 11/15/2020, Auckland,NZ
#Loss function
#Reference: https://en.wikipedia.org/wiki/Loss_function
#https://papers.nips.cc/paper/2008/file/f5deaeeae1538fb6c45901d524ee2f98-Paper.pdf
import numpy as np

def Least_squares(x): #LS
    return (1-x)**2

def Modified_LS(x):
    def loss(x):
        return (np.max(1-x, 0))**2
    return list(map(loss, list(x)))

def SVM_Loss(x): #Hinge loss, SVM using
    def loss(x):
        return np.max(1-x, 0)
    return list(map(loss, list(x)))

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

#Cross-Entropy loss L = -(y*log(y') + (1-y)*log(1-y'))
def crossEntropy_GT01(y, yPred): # 0/1 classification, y:Ground truth: 0 or 1, yPred:0~1
    if y == 0:
        return -1*np.log(1-yPred)
    return -1*np.log(yPred) #y=1

#Cross-Entropy loss L = log(1 + e^(-y*y')) , activefun=sigmoid()
def crossEntropy_GT02(y, yPred): # +1/1 classification, y:Ground truth:-1 or +1, yPred:-1~1
    if y == 1:
        return np.log(1 + np.exp(-1*yPred))
    return np.log(1 + np.exp(yPred)) #y=-1

#Cross-Entropy(CE) Loss: CE(p) = -log(p), when y=1
#wighted Cross-Entropy, like Focal loss(FC), FC(p) = -(1-p)^gamma*log(p)
# FC == CE,when gamma == 0
def FocalLosss(p, gamma=0):
    return -np.power(1-p, gamma)*np.log(p) #return -gamma*np.log(p)

#CE(p) = -log(1-p), when y=0
#FC(p) = -(p)^gamma*log(1-p)
def FocalLosss1(p, gamma=0):
    return -np.power(p, gamma)*np.log(1-p) #return -gamma*np.log(p)

def TripletLoss(x, m=0.2):#https://en.wikipedia.org/wiki/Triplet_loss
    #max(t_p - t_n + m, 0)
    def loss(t):
        return max(t+m, 0)
    return list(map(loss, list(x)))

def ContrastiveLoss(x, m=0.2):
    #max(t_p - t_n + m, 0)
    def loss(t):
        return max(m-t, 0)
    return list(map(loss, list(x)))

def BinomialDevianceLoss(x):
    return np.log(1+np.exp(-2*x))

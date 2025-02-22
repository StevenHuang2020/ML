#steven 21/03/2020
#01/04/2020  add derivative2
#common and interesting math fuction

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from activationFunc import *
from lossFunc import *
from functions import *
from plotCommon import *
from KL_divergence import get_klpq_div,get_klqp_div
from distributions import Discrete_uniform_distribution,Binomial_distribution
import mpl_toolkits.axisartist as axisartist

import matplotlib
matplotlib.rcParams['savefig.dpi'] = 300


def testLogisticMap():
    rValue = 3.9
    ls1 = logisticMap(r=rValue,x0=0.2,N=100)
    #plot(ls1)
    ls2 = logisticMap(r=rValue,x0=0.20000000001,N=100)
    #plot(ls2)

    ax = plt.subplot(1, 1, 1)
    ax.plot(ls1,label='x0=0.2')
    ax.plot(ls2,label='x0=0.20000000001')

    xMax = 100
    ax.set_xticks(np.linspace(0,xMax,10))
    ax.set_xlim(0, xMax)
    ax.set_title("An iota causes a big difference(butterfly effects)")
    ax.legend(loc='lower left')
    plt.show()


def testLogisticMap2():
    #rValue = 9
    #ls1 = logisticMap2(r=rValue,x0=0.2,N=100)
    #plot(ls1)
    #ls2 = logisticMap(r=rValue,x0=0.20000000001,N=100)
    #plot(ls2)

    #x = np.linspace(-1,1, 10000)
    #scatter(x,circle(x))
    #scatter(x,heart(x))
    #Num=100
    #scatter(np.arange(Num),logisticMap(r=.5,x0=0.4,N=Num),False)
    #plot(logisticMap(r=.5,x0=0.4,N=30))

    '''
    N=9
    for i in range(N):
        print('*'*50,i)
        rValue = i+1
        ls1 = logisticMap2(r=rValue,x0=0.2,N=100)

        ax = plt.subplot(3, 3, i+1)
        ax.plot(ls1,label='x0=0.2')
        ax.set_title('r='+str(rValue))
    plt.show()
    '''

    iters=80
    rStart = 3.9
    #rl=[0, 0.01, 0.001, 0.0001, 0.0000001, 0.0000000001]
    rl=[0, 0.000001, 0.0000000000001]
    ls=[]
    for i,rV in enumerate(rl):
        #print('*'*50,i)
        rValue = rStart + rV
        ls.append(logisticMap(r=rValue,x0=0.8,N=iters))

    ax = plt.subplot(1,1,1)
    for i,l in enumerate(ls):
        ax.plot(l,label='x0='+str(rStart + rl[i]))

    ax.set_title('Butterfly effects')
    ax.legend(loc='lower left')
    plt.show()

def plotCircle(ax):
    x = np.linspace(-1.0, 1.0, 100)
    y = np.linspace(-1.0, 1.0, 100)
    X, Y = np.meshgrid(x, y)
    F = X ** 2 + Y ** 2 - 1
    ax.contour(X, Y, F, [0])
    ax.set_aspect(1)

def plotHeart(ax):
    x = np.linspace(-1.0, 1.0, 100)
    y = np.linspace(-1.0, 1.5, 100)
    X, Y = np.meshgrid(x, y)
    # x**2+ (5*y/4 - sqrt(abs(x)))**2 = 1
    F = X ** 2 + (5*Y/4 - np.sqrt(abs(x)))** 2 - 1
    ax.contour(X, Y, F, [0])
    ax.set_aspect(1)

def plotPowerX():
    x = np.linspace(0,1.5, 100)
    ax = plt.subplot(1,1,1)
    plotSub(x, powerX(x), ax,label='powerX')
    #plotSub(x, derivative(powerX, x), ax, label='powerX\'', aspect=True)

    de = derivative(powerX, x)
    de[x<0.1]= np.nan
    plotSub(x, de, ax, label='powerX\'', aspect=False)

    #plotSub(x, derivative2(powerX, x), ax, label='powerX\'\'', aspect=True)
    plotSub(x, x, ax,label='y=x')
    plotSub(x, np.ones((len(x))), ax,label='y=1')
    plt.show()

def plotAllFuc():
    x = np.linspace(-2,2, 50)
    ax = plt.subplot(1,1,1)

    plotSub(x, sigmoid(x), ax,label='sigmoid')
    plotSub(x, derivative(sigmoid, x), ax, label='sigmoid\'', aspect=False)
    plotSub(x, derivative2(sigmoid, x), ax, label='sigmoid\'\'', aspect=False)

    plotSub(x, normalDistribution(x), ax,label='normalDistribution')
    plotSub(x, derivative(normalDistribution, x), ax, label='normalDistribution\'', aspect=False)
    plotSub(x, derivative2(normalDistribution, x), ax, label='normalDistribution\'\'', aspect=False)

    #plotSub(x, exp(x), ax,name='exp', label='exp')
    #plotSub(x, derivative(exp, x), ax, label='exp\'', aspect=False)

    plotSub(x, softmaxFuc(x), ax,label='softmaxFuc')
    plotSub(x, derivative(softmaxFuc, x), ax, label='softmaxFuc\'', aspect=False)

    #plotSub(x, log(x), ax,label='log') #log(x)' = 1/x
    #plotSub(x, derivative(log, x), ax,label='log\'')

    plotHeart(ax)
    plotCircle(ax)
    plt.show()

def plotSoftmax():
    x = np.linspace(-2,-1, 100)
    ax = plt.subplot(1,1,1)
    plotSub(x, softmaxFuc(x), ax,label='softmaxFuc[-2,-1]')

    x = np.linspace(-1,0, 100)
    plotSub(x, softmaxFuc(x), ax,label='softmaxFuc[-1,0]')

    x = np.linspace(-0.5,0.5, 100)
    plotSub(x, softmaxFuc(x), ax,label='softmaxFuc[-0.5,0.5]')

    x = np.linspace(0,1, 100)
    plotSub(x, softmaxFuc(x), ax,label='softmaxFuc[0,1]')

    x = np.linspace(1,3, 100)
    plotSub(x, softmaxFuc(x), ax,label='softmaxFuc[1,3]')

    x = np.linspace(3,5, 100)
    plotSub(x, softmaxFuc(x), ax,label='softmaxFuc[3,5]')

    x = np.linspace(-1,3, 100)
    plotSub(x, softmaxFuc(x), ax,label='softmaxFuc[-1,3]')

    x = np.linspace(-2,5, 100)
    plotSub(x, softmaxFuc(x), ax,label='softmaxFuc[-2,5]')

    plt.show()

def plotSoftmax2():
    ax = plt.subplot(1,1,1)
    Ns=[10,20,50,100,1000]
    for i in Ns:
        x = np.linspace(0,1, i)
        plotSub(x, softmaxFuc(x), ax,label='softmaxFuc[0,1]-'+str(i))
    plt.show()

def plotActivationFucBRelu():
    ax = plt.subplot(1,1,1)
    x = np.linspace(-0.1,0.1, 40)
    ax.set_title('Activation BRelu')
    plotSub(x, BReLu(x), ax,label='BReLu-even')
    y = np.linspace(-0.1,0.1, 41)
    plotSub(y, BReLu(y), ax,label='BReLu-odd')
    plt.show()

def plotActivationOneFun():
    ax = plt.subplot(1,1,1)
    x = np.linspace(-20,20, 100)
    ax.set_title('Activation')

    #plotSub(x, softPlus(x), ax,label='softPlus')
    #plotSub(x, BentIdentity(x), ax,label='BentIdentity')
    #plotSub(x, SoftClipping(x), ax,label='SoftClipping')

    #plotSub(x, SoftExponential(x), ax,label='SoftExponential')
    #plotSub(x, Sinusoid(x), ax,label='Sinusoid')
    #plotSub(x, Sinc(x), ax,label='Sinc')
    #plotSub(x, Gaussian(x), ax,label='Gaussian')

    x = np.linspace(-5,5, 100)
    plotSub(x, SQ_RBF(x), ax,label='SQ_RBF')
    plt.show()

def plotEquationSlove():
    ax = plt.subplot(1,1,1)
    x = np.linspace(-4,5, 100)
    ax.set_title('x^2 = 2^x, not just 2 and 4')
    plotSub(x, x**2, ax,label='x^2')
    plotSub(x, powerAX(2,x), ax,label='2^x')
    plt.show()

def plotActivationFunSoftE():
    ax = plt.subplot(1,1,1)
    x = np.linspace(-2,2, 50)
    ax.set_title('Activation SoftExponential')

    alphas = np.linspace(-1,1, 9)
    for i in alphas:
        plotSub(x, SoftExponential(x,i), ax,label='SoftExponential_alpha('+str(i)+')')
    plt.show()

def plotActivationFunSquashing():
    ax = plt.subplot(1,1,1)
    x = np.linspace(-2,2, 100)
    ax.set_title('Activation Squashing')
    plotSub(x, Squashing(x,beta=1), ax,label='Squashing_beta=1')
    plotSub(x, Squashing(x,beta=2), ax,label='Squashing_beta=2')
    plotSub(x, Squashing(x,beta=5), ax,label='Squashing_beta=5')
    plotSub(x, Squashing(x,beta=50), ax,label='Squashing_beta=50')
    plt.show()

def plotMyActivationFun():
    ax = plt.subplot(1,1,1)

    x = np.linspace(-2,2, 80)
    ax.set_title('') #Activation Function

    #plotSub(x, ISRu(x), ax,label='ISRu')
    #plotSub(x, ISRLu(x), ax,label='ISRLu')
    #plotSub(x, PLu(x), ax,label='PLu')

    plotSub(x, Relu(x), ax,label='$ReLU$')
    #plotSub(x, derivative(Relu, x), ax, label="$ReLU^{'}$", aspect=False)

    #plotSub(x, LeakyRelu(x), ax,label='LeakyReLU')
    #plotSub(x, PRelu(x), ax,label='PRelu')
    #plotSub(x, RRelu(x), ax,label='RRelu')
    plotSub(x, GELu(x), ax,label='$GELU$')
    #plotSub(x, derivative(GELu, x), ax, label="$GELU^{'}$", aspect=False)
    #plotSub(x, ELu(x), ax,label='ELu')
    #plotSub(x, SELu(x), ax,label='SELu')

    # plotSub(x, BReLu(x), ax,label='BReLu-even')
    # y = np.linspace(-2,2, 51)
    # plotSub(y, BReLu(y), ax,label='BReLu-odd')

    plt.ylim(-0.4, 2.1)
    #plt.legend()
    plt.grid(linestyle='-.') #'-', '--', '-.', ':', '',
    plt.legend(ncol=4,loc='upper left')
    plt.show()

def plotActivationFun():
    ax = plt.subplot(1,1,1)

    x = np.linspace(-2,2, 80)
    ax.set_title('Activation Function')

    plotSub(x, Identity(x), ax,label='Identity')
    plotSub(x, Binary_step(x), ax,label='Binary_step')
    plotSub(x, sigmoid(x), ax,label='sigmoid')
    plotSub(x, Tanh(x), ax,label='Tanh')
    plotSub(x, SQNL(x), ax,label='SQNL')
    plotSub(x, ArcTan(x), ax,label='ArcTan')
    plotSub(x, ArcSinH(x), ax,label='ArcSinH')

    plotSub(x, Softsign(x), ax,label='Softsign')
    plotSub(x, ISRu(x), ax,label='ISRu')
    plotSub(x, ISRLu(x), ax,label='ISRLu')
    plotSub(x, PLu(x), ax,label='PLu')
    plotSub(x, Relu(x), ax,label='Relu')

    plotSub(x, LeakyRelu(x), ax,label='LeakyRelu')
    plotSub(x, PRelu(x), ax,label='PRelu')
    plotSub(x, RRelu(x), ax,label='RRelu')
    plotSub(x, GELu(x), ax,label='GELu')
    plotSub(x, ELu(x), ax,label='ELu')
    plotSub(x, SELu(x), ax,label='SELu')

    plotSub(x, BReLu(x), ax,label='BReLu-even')
    y = np.linspace(-2,2, 51)
    plotSub(y, BReLu(y), ax,label='BReLu-odd')

    plotSub(x, softPlus(x), ax,label='softPlus')
    plotSub(x, BentIdentity(x), ax,label='BentIdentity')
    plotSub(x, SoftClipping(x), ax,label='SoftClipping')
    plotSub(x, Squashing(x,beta=1), ax,label='Squashing_beta=1')
    plotSub(x, SoftExponential(x), ax,label='SoftExponential')
    plotSub(x, Sinusoid(x), ax,label='Sinusoid')
    plotSub(x, Sinc(x), ax,label='Sinc')
    plotSub(x, Gaussian(x), ax,label='Gaussian')
    plotSub(x, SQ_RBF(x), ax,label='SQ_RBF')
    plotSub(x, swish(x), ax,label='swish')

    plt.ylim(-2, 6)
    #plt.legend()
    plt.legend(ncol=4,loc='upper left')
    plt.show()


def plotAxis():

    #创建画布
    fig = plt.figure(figsize=(8, 8))
    #使用axisartist.Subplot方法创建一个绘图区对象ax
    ax = axisartist.Subplot(fig, 111)
    #将绘图区对象添加到画布中
    fig.add_axes(ax)

    ax.axis[:].set_visible(False)#通过set_visible方法设置绘图区所有坐标轴隐藏
    ax.axis["x"] = ax.new_floating_axis(0,0)#ax.new_floating_axis代表添加新的坐标轴
    ax.axis["x"].set_axisline_style("->", size = 1.0)#给x坐标轴加上箭头
    #添加y坐标轴，且加上箭头
    ax.axis["y"] = ax.new_floating_axis(1,0)
    ax.axis["y"].set_axisline_style("-|>", size = 1.0)
    #设置x、y轴上刻度显示方向
    ax.axis["x"].set_axis_direction("top")
    ax.axis["y"].set_axis_direction("right")

    x = np.arange(-15,15,0.1)
    #生成sigmiod形式的y数据
    y=1/(1+np.exp(-x))
    #设置x、y坐标轴的范围
    plt.xlim(-12,12)
    plt.ylim(0, 1)
    #绘制图形
    plt.plot(x,y, c='b')
    plt.grid(linestyle='-.')
    plt.show()


def plotCommonActivationFun():
    ax = plt.subplot(1,1,1)

    N=50
    if 0:
        x = np.linspace(-4,4, N)
        title = 'Sigmoid'
        plotSub(x, sigmoid(x), ax)
    elif 0:
        x = np.linspace(-4,4, N)
        title = 'Tanh'
        plotSub(x, Tanh(x), ax)
    else:
        x = np.linspace(-2,2, N)
        title = 'ReLU'
        plotSub(x, Relu(x), ax)

    ax.set_title(title)

    #plt.ylim(-2, 6)
    #plt.legend()
    #plt.legend(ncol=4,loc='upper left')
    plt.grid(linestyle='-.')
    plt.savefig(r'.\res\Activation.png', dpi=300)
    plt.show()

def plotLogX():
    ax = plt.subplot(1,1,1)
    x = np.linspace(0,1, 50)
    ax.set_title('Log Function')

    plotSub(x, log(x), ax,label='log(x)')
    plotSub(x, log(1-x), ax,label='log(1-x)')
    plotSub(x, -1*log(1-x), ax,label='-log(1-x)')

    plt.legend()
    plt.show()

def plotLogEntropy():
    ax = plt.subplot(1,1,1)
    x = np.linspace(-2,2, 50)
    ax.set_title('LogEntropy Function')

    #plotSub(x, log(x), ax,label='log')
    #plotSub(x, normalDistribution(x), ax,label='normal')
    p = normalDistribution(x)
    plotSub(x, entropy(p), ax,label='entropy')
    plt.legend()
    plt.show()

def plotKL_Divergences():
    ax = plt.subplot(1,1,1)
    x = np.linspace(-6,6, 100)
    ax.set_title('KL_Divergence')

    #plotSub(x, log(x), ax,label='log')
    #plotSub(x, normalDistribution(x), ax,label='normal')
    p = NormalDistribution(x)
    q = NormalDistribution(x,u=1)
    plotSub(x, p, ax,label='normal,u=0')
    plotSub(x, q, ax,label='normal,u=1')

    print('kl_divergence=',KL_divergence(0))
    y = list(map(KL_divergence, list(x)))
    #print('y=',y)

    ypq=[ i[0] for i in y]
    yqp=[ i[1] for i in y]

    #print('yqp=',yqp)
    plotSub(x, ypq, ax,label='kl_divergencePQ')
    plotSub(x, yqp, ax,label='kl_divergenceQP')
    plt.vlines(0, 0, 0.42,linestyles='dotted') #'solid', 'dashed', 'dashdot', 'dotted'
    plt.hlines(0, -6, 6,linestyles='dotted')
    plt.legend()
    plt.show()

def plot_gamma():
    ax = plt.subplot(1,1,1)
    x = np.linspace(-10,7, 1000)
    ax.set_title('gamma Function')
    plotSub(x, gamma(x), ax,label='gamma(x)')

    k = np.arange(1, 8)
    scatterSub(k, factorial(k-1),ax, label='(x-1)!, x = 1, 2, ...', marker='*')

    plt.xlim(-7.5, 8.5)
    plt.ylim(-40, 160)
    plt.grid()
    plt.xlabel('x')
    plt.legend(loc='lower right')
    plt.show()

def yieldBetaFuc(start,stop,N):
    xvalues = np.linspace(start,stop,N)
    yvalues = np.linspace(start,stop,N)
    for u, x in enumerate(xvalues):
        for v, y in enumerate(yvalues):
            #print('x,y=',x,y,beta(x,y))
            yield u,v,beta(x,y)

def getBetaFucImg(start,stop,N):
    M = np.zeros([N, N,3], int) # + 255
    for v,u,z in yieldBetaFuc(start,stop,N): #map z(0~2) to 0~255 pixsel value
        value =  int(z*256/.2)
        #print('z=',v,u,z,value)
        #M[v, u, :] = value
        #M[v, u, 0] = value #r channel
        M[v, u, 1] = value #g channel
        #M[v, u, 2] = value #b channel
    return M

def plotBetaFuc2():
    start=1
    stop =3
    N=1000
    x = np.linspace(start,stop, N)
    plt.imshow(getBetaFucImg(start,stop,N),cmap='gray')
    plt.show()

def plot_beta():
    ax = plt.subplot(1,1,1)
    x = np.linspace(-3,3, 200)
    print(x.shape)
    x,y = np.meshgrid(x,x)
    v = beta(x,y).flatten()
    print(v.shape)
    print(x.shape)
    print(y.shape)
    #return
    ax.set_title('beta Function')
    #plotSub(x, y, ax,label='beta(x)')
    #scatterSub(x,beta(x,y),ax, label='beta', marker='.')
    scatterSub(x,y,ax, c=v)

    #plt.grid()
    plt.xlabel('x')
    plt.legend(loc='lower right')
    plt.show()

def plotHeart():
    row = 2
    col = 2
    ax = plt.subplot(row, col, 1)
    x = np.linspace(-5,5, 10)
    plotSub(x,sigmoid(x), ax,label='sigmod')

    ax = plt.subplot(row, col, 2)
    x = np.linspace(-5,0, 10)
    plotSub(x,exp(x), ax,label='exp')

    ax = plt.subplot(row, col, 3)
    x = np.linspace(-1,1, 1000)
    scatterSub(x,circle(x), ax,label='circle')

    ax = plt.subplot(row, col, 4)
    scatterSub(x,heart(x), ax,label='heart')
    plt.show()

def plotDivisorfunction(N=200):
    ax = plt.subplot(1,1,1)
    ax.set_title('Divisor function')

    x = np.arange(1,N)
    y = list(map(Divisorfunction, [(i,0) for i in x]))
    plotSub(x, y, ax,label='N=250,p=0')

    y1 = list(map(Divisorfunction, [(i,1) for i in x]))
    plotSub(x, y1, ax,label='N=250,p=1')

    #y2 = list(map(Divisorfunction, [(i,2) for i in x]))
    #plotSub(x, y2, ax,label='N=250,p=2')
    plt.show()

def plotEulerTotients():
    ax = plt.subplot(1,1,1)
    ax.set_title('Euler Totients')
    N=500
    x = np.arange(1,N)
    y = list(map(EulerTotients, list(x)))
    #print('y=',y)
    scatterSub(x,y,ax,label='totients')
    plt.show()

def plotPrimeNumbers():
    ax = plt.subplot(1,1,1)
    ax.set_title('Prime counter Function')

    N=60
    x = np.arange(1,N)
    y = list(map(PrimeNumbers, list(x)))
    scatterSub(x,y,ax,label='Prime counter')

    x = np.linspace(0,N, 100)
    y = x/np.log(x)
    plotSub(x, y, ax,label='x/ln(x)')

    ax.legend()
    plt.show()

def plotMobiusfunction():
    ax = plt.subplot(1,1,1)
    ax.set_title('Mobius function')

    N=60
    x = np.arange(1,N)
    y = list(map(Mobiusfunction, list(x)))
    print(x)
    print(y)
    scatterSub(x,y,ax,label='Mobius')
    plt.show()

def plotLegendreFunction():
    ax = plt.subplot(1,1,1)
    ax.set_title('Legendre Function')

    x = np.linspace(-1,1, 200)
    plotSub(x, LegendreFunction(x), ax,label='Q0')
    plotSub(x, LegendreFunction(x,n=1), ax,label='Q1')
    plotSub(x, LegendreFunction(x,n=2), ax,label='Q2')
    plotSub(x, LegendreFunction(x,n=3), ax,label='Q3')
    plotSub(x, LegendreFunction(x,n=4), ax,label='Q4')
    ax.legend()
    plt.show()

def plotScorersFunction():
    ax = plt.subplot(1,1,1)
    ax.set_title('Scorers Function')

    x = np.linspace(-10,8, 50)
    yGi=[]
    yHi=[]
    for _ in x:
        gi,hi =ScorersFunctionGi(_)
        yGi.append(gi)
        yHi.append(hi)

    plotSub(x, yGi, ax,label='Gi')
    plotSub(x, yHi, ax,label='Hi')
    #plotSub(x, LegendreFunction(x,n=1), ax,label='Q1')

    ax.legend()
    plt.show()

def plotLogarithmic_integral():
    ax = plt.subplot(1,1,1)
    ax.set_title('Logarithmic_integral Function')

    x1 = [] #np.linspace(0,1, 50)
    y1=[]
    #for _ in x1:
        #y1.append(Logarithmic_integral(_))

    x2 = np.linspace(1,2, 50)
    y2=[ Logarithmic_integral(_) for _ in x2 ]

    x = np.concatenate((x1,x2),axis=0)
    y = np.concatenate((y1,y2),axis=0)
    plotSub(x, y, ax,label='Li')
    ax.legend()
    plt.show()

def plotExponential_integral():
    ax = plt.subplot(1,1,1)
    ax.set_title('Exponential_integral Function')

    x = np.linspace(0,4, 50)
    y=[ Exponential_integral(_) for _ in x]

    plotSub(x, y, ax,label='Exponential_integral ')
    plt.show()

def plotTrigonometric_integral():
    ax = plt.subplot(1,1,1)
    ax.set_title('Trigonometric_integral Function')

    x = np.linspace(0,25, 200)
    ySi=[]
    yCi=[]
    for _ in x:
        si,ci =Trigonometric_integral(_)
        ySi.append(si)
        yCi.append(ci)

    plotSub(x, ySi, ax,label='Si')
    plotSub(x, yCi, ax,label='Ci')
    ax.legend()
    plt.show()

def plot_LossFunctions():
    ax = plt.subplot(1,1,1)
    ax.set_title('Loss Functions')

    x = np.linspace(-4,2, 100)
    plotSub(x, Least_squares(x), ax, label='Least_squares')
    plotSub(x, Modified_LS(x), ax, label='Modified_LS', linestyle='dashed')
    plotSub(x, SVM_Loss(x), ax, label='SVM_Loss')
    plotSub(x, Boosting_Loss(x), ax, label='Boosting_Loss')
    plotSub(x, LogisticRegression(x), ax, label='LogisticRegression')
    plotSub(x, Savage_loss(x), ax, label='Savage_loss', linestyle='dashdot')
    plotSub(x, zero_one(x), ax, label='zero_one')

    ax.legend()
    plt.ylim(0, 4.5)
    plt.show()

def plot_LossFunctions2(type=0):
    ax = plt.subplot(1,1,1)

    if type==0:
        x = np.linspace(0,1, 100)

        title='Loss Functions Cross binary entropy 0/1'
        ax.set_title(title)
        plotSub(x, crossEntropy_GT01(0, x), ax, label="L=-(y*log(y')+(1-y)*log(1-y')) when y=0")
        plotSub(x, crossEntropy_GT01(1, x), ax, label="L=-(y*log(y')+(1-y)*log(1-y')) when y=1")
    else:
        x = np.linspace(-1,1, 100)

        title='Loss Functions Cross binary entropy -1/1'
        ax.set_title(title)
        plotSub(x, crossEntropy_GT02(1, x), ax, label="L=log(1+e^(-y*y')) when y=1")
        plotSub(x, crossEntropy_GT02(-1, x), ax, label="L=log(1+e^(-y*y')) when y=-1")

    ax.legend()
    #plt.ylim(0, 4.5)
    plt.show()

def plot_LossFocalLoss():
    ax = plt.subplot(1,1,1)
    #ax.set_title('Focal Loss Function')

    x = np.linspace(0.0001, 0.99999, 100)

    #plt.text(0.12, 4.5, r'$CE(p_t)=-\log(p_t)$')
    #plt.text(0.12, 4.2, r'$FC(p_t)=-(1-p_t)^\gamma\log(p_t)$')
    plt.text(0.12, 4.5, r'$BCE(\hat{y},y)=-\log(\hat{y})$')
    plt.text(0.12, 4.2, r'$FL(\hat{y},y)=-(1-\hat{y})^\gamma\log(\hat{y})$')

    plotSub(x, FocalLosss(x), ax, label=r'BCE, $\gamma=0$')
    plotSub(x, FocalLosss(x,gamma=0.5), ax, label=r'$\gamma=0.5$', linestyle='dashed')
    plotSub(x, FocalLosss(x,gamma=1), ax, label=r'$\gamma=1$')
    plotSub(x, FocalLosss(x,gamma=2), ax, label=r'$\gamma=2$', linestyle='dotted') #dashdot
    plotSub(x, FocalLosss(x,gamma=5), ax, label=r'$\gamma=5$')

    ax.legend()
    plt.xlabel('Probability of truth sample')
    plt.ylabel('Loss')
    plt.ylim(0, 5)
    plt.subplots_adjust(left=0.085, bottom=None, right=0.98, top=0.95, wspace=None, hspace=None)
    plt.show()

def plot_LossFocalLoss1():
    ax = plt.subplot(1,1,1)
    #ax.set_title('Focal Loss Function')

    x = np.linspace(0.0001, 0.99999, 100)

    #CE(p) = -log(1-p), when y=0
    #FC(p) = -(p)^gamma*log(1-p)

    # plt.text(0.46, 4.5, r'$CE(p_t)=-\log(1-p_t)$')
    # plt.text(0.46, 4.2, r'$FC(p_t)=-(p_t)^\gamma\log(1-p_t)$')
    plt.text(0.36, 4.5, r'$BCE(\hat{y},y)=-\log(1-\hat{y})$')
    plt.text(0.36, 4.2, r'$FL(\hat{y},y)=-(\hat{y})^\gamma\log(1-\hat{y})$')

    plotSub(x, FocalLosss1(x), ax, label=r'BCE, $\gamma=0$')
    plotSub(x, FocalLosss1(x,gamma=0.5), ax, label=r'$\gamma=0.5$', linestyle='dashed')
    plotSub(x, FocalLosss1(x,gamma=1), ax, label=r'$\gamma=1$')
    plotSub(x, FocalLosss1(x,gamma=2), ax, label=r'$\gamma=2$', linestyle='dotted') #dashdot
    plotSub(x, FocalLosss1(x,gamma=5), ax, label=r'$\gamma=5$')

    ax.legend()
    plt.xlabel('Probability of false sample')
    plt.ylabel('Loss')
    plt.ylim(0, 5)
    plt.subplots_adjust(left=0.085, bottom=None, right=0.98, top=0.95, wspace=None, hspace=None)
    plt.show()

def plot_TripletLoss():
    ax = plt.subplot(1,1,1)
    #ax.set_title('Focal Loss Function')

    x = np.linspace(-2, 2, 100)

    plotSub(x, TripletLoss(x), ax, label=r'$\max(x+m,0)$')
    plotSub(x, ContrastiveLoss(x), ax, label=r'$\max(x-m,0)$')
    plotSub(x, BinomialDevianceLoss(x), ax, label=r'$\max(x-m,0)$')

    ax.legend()
    # plt.xlabel('Probability of truth sample')
    # plt.ylabel('Loss')
    # plt.ylim(0, 5)
    #plt.subplots_adjust(left=0.085, bottom=None, right=0.98, top=0.95, wspace=None, hspace=None)
    plt.show()

def plot_KLdivergence_Loss():
    ax = plt.subplot(1,1,1)
    ax.set_title('KLdivergence Loss')

    N=100
    x = np.linspace(0.0001, 0.9999, N)
    target = Binomial_distribution(N,0.7) #Discrete_uniform_distribution(x,N)
    #print(x.shape,target.shape)
    #plt.text(0.12, 4.5, r'$CE(p)=-\log(p)$')
    klpq = [get_klpq_div(Binomial_distribution(N,i),target) for i in x]
    klqp = [get_klpq_div(target, Binomial_distribution(N,i)) for i in x]

    #y = [get_klpq_div(np.zeros_like(target)+i, target) for i in x]
    #y = [get_klqp_div(np.zeros_like(target)+i, target) for i in x]
    plotSub(x, klpq, ax, label=r'$KL(p,q)=p*\log(\frac{p}{q})$')
    plotSub(x, klqp, ax, label=r'$KL(q,p)=q*\log(\frac{q}{p})$')

    ax.legend()
    #plt.ylim(0, 5)
    plt.show()

def plotACON_Activation():
    ax = plt.subplot(1,1,1)
    ax.set_title('ACON-C activation function')

    x = np.linspace(-6, 6, 100)

    plt.text(-5, 6.5, r'$ACON-C = (p1-p2)*x*\delta(\beta*(p1-p2)*x)+p2*x$')
    plt.text(-5, 6, r'$where$')
    plt.text(-3.6, 6, r'$\delta = sigmoid(x)$')
    #plt.text(0.36, 4.2, r'$FL(\hat{y},y)=-(\hat{y})^\gamma\log(1-\hat{y})$')

    p1 = 1.2
    p2 = -0.8
    plotSub(x, ACON_C(x,p1,p2, beta=0.01), ax, label=r'$\beta=0.01$')
    plotSub(x, ACON_C(x,p1,p2, beta=0.1), ax, label=r'$\beta=0.1$')
    plotSub(x, ACON_C(x,p1,p2, beta=1), ax, label=r'$\beta=1$')


    ax.legend(loc='lower right')
    #plt.xlabel('')
    plt.ylabel('ACON(p1x, p2x)')
    #plt.ylim(0, 5)
    #plt.subplots_adjust(left=0.085, bottom=None, right=0.98, top=0.95, wspace=None, hspace=None)
    plt.show()

def plotAlgorithmBigO(): #algorithm complexity
    plt.figure(figsize=(12,10))
    ax = plt.subplot(1,1,1)
    ax.set_title('Algorithm complexity')

    n = np.linspace(1,10)

    #lables=['Constant', 'Logarithmic', 'Linear', 'Log Linear', 'Quadratic', 'Cubic', 'Exponential']
    lables=['$Constant=1$', '$Logarithmic=\log(n)$', '$Linear=n$', \
        '$Log Linear=n*\log(n)$', '$Quadratic=n^2$', '$Cubic=n^3$', '$Exponential=2^n$']
    big_o=[np.ones(n.shape), np.log(n), n, n*np.log(n), n**2, n**3, 2**n]

    #plt.text(-5, 6.5, r'$ACON-C = (p1-p2)*x*\delta(\beta*(p1-p2)*x)+p2*x$')
    #plt.text(-5, 6, r'$where$')

    for i,label in enumerate(lables):
        plotSub(n, big_o[i], ax, label=label)

    ax.legend(loc=0)
    plt.xlabel('n')
    plt.ylabel('Relateive Runtime')
    plt.ylim(0, 50)
    #plt.subplots_adjust(left=0.085, bottom=None, right=0.98, top=0.95, wspace=None, hspace=None)
    plt.show()

def main():
    #return testLogisticMap2()
    #plotAllFuc()
    #plotPowerX()
    #plotSoftmax()
    #plotSoftmax2()
    #plotActivationFun()
    #plotActivationFunSoftE()
    #plotActivationFucBRelu()
    #plotActivationFunSquashing()
    #plotActivationOneFun()
    #plotEquationSlove()
    #plotLogEntropy()
    #plotLogX()
    #plot_gamma()
    #plot_beta()
    #plotBetaFuc2()
    #plotDivisorfunction()
    #plotHeart()
    #plotEulerTotients()
    #plotPrimeNumbers()
    #plotMobiusfunction()
    #plotLegendreFunction()
    #plotScorersFunction()
    #plotLogarithmic_integral()
    #plotExponential_integral()
    #plotTrigonometric_integral()
    #plotKL_Divergences()
    #plot_LossFunctions()
    #plot_LossFunctions2(type=0)
    #plot_KLdivergence_Loss()
    #plot_LossFocalLoss()
    #plot_LossFocalLoss1()
    #plotMyActivationFun()
    #plot_TripletLoss()

    #plotAxis()
    #plotCommonActivationFun()
    #plotACON_Activation()
    plotAlgorithmBigO()

if __name__ == '__main__':
    main()

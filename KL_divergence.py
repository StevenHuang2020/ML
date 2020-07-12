import matplotlib.pyplot as plt
import numpy as np
from distributions import *

def plotDistribute(ax, data, label='', width=0.3, offset=0, title='Probability Distribution of true'):
    ax.bar(np.arange(len(data))+offset,data,width=width,label=label)
    
    fontSize = 12
    ax.set_title(title,fontsize=fontSize)

    plt.xlabel('Different Teeth Bins',fontsize=fontSize)
    plt.ylabel('Probability',fontsize=fontSize)
    plt.xticks(np.arange(len(data)))
    
def plorDataDis(true_data,uniform,bino):
    ax = plt.subplot(1,1,1)
    offset = 0
    width=0.2
    plotDistribute(ax, true_data, label='True data', offset=offset, width=width)
    
    offset += width
    plotDistribute(ax, uniform, label='Uniform data', offset=offset, width=width)
    
    offset += width
    print(bino)
    plotDistribute(ax, bino, label='Binomial data', offset=offset, width=width)
    plt.legend()
    plt.show()
    
def get_klpq_div(p_probs, q_probs):
    kl_div = 0.0
    for pi, qi in zip(p_probs, q_probs):
        kl_div += pi*np.log(pi/qi)

    return kl_div

def get_klqp_div(p_probs, q_probs):
    kl_div = 0.0
    for pi, qi in zip(p_probs, q_probs):
        kl_div += qi*np.log(qi/pi)
    
    return kl_div


def testDiscretKL():
    true_data = [0.02, 0.03, 0.15, 0.14, 0.13, 0.12, 0.09, 0.08, 0.1, 0.08, 0.06]
    print('sum=', sum(true_data))
    assert sum(true_data)==1.0
    unif_data = Discrete_uniform_distribution(true_data,N=len(true_data))
    bino_data = Binomial_distribution(N=len(true_data),p=0.3)
    
    #plorDataDis(true_data,unif_data,bino)
    
    print('KL(True||Uniform): ', get_klpq_div(true_data,unif_data))
    print('KL(True||Binomial): ', get_klpq_div(true_data,bino_data))
    
    p = np.arange(0.02, 1.0, 0.02) #np.linspace(0, 1.0, 50)
    klpq = [get_klpq_div(true_data,Binomial_distribution(N=len(true_data),p=i)) for i in p]
    klqp = [get_klqp_div(true_data,Binomial_distribution(N=len(true_data),p=i)) for i in p]

    print('minimal klpq,', np.argmin(klpq), np.min(klpq))
    
    ax = plt.subplot(1,1,1)
    plotDistribute(ax,p,klpq,label='KL(P||Q)')
    plotDistribute(ax,p,klqp,label='KL(Q||P)')
    plotDistribute(ax,p,np.array(klpq)-np.array(klqp),label='KL(P||Q)-KL(Q||P)')
    plt.show()
    
def plotDistribute(ax,x,y,label='', title='Binomial P vs KL'):
    ax.plot(x,y,label=label)
    
    fontSize = 12
    ax.set_title(title,fontsize=fontSize)
    ax.legend()
    plt.xlabel('Binomial P',fontsize=fontSize)
    plt.ylabel('KL(P||Q) divergence',fontsize=fontSize)
    #plt.show()
    
def main():
    testDiscretKL()
    
if __name__=='__main__':
    main()
    
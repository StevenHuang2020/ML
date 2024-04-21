#python3 steven
#plot logistic map equation
#https://en.wikipedia.org/wiki/Logistic_map
#logistic map equation:  x := r*x*(1-x)

import numpy as np
import matplotlib.pyplot as plt

# Logistic function implementation
def logistic_eq(r,x):
    #return r*x*(1-x)
    #return r*x*(1-x)**2
    return r*x*(1-x)**3
    #return r*x*(1-x)**5

# Iterate the function for a given r
def logistic_equation_orbit(seed, r, n_iter, n_skip=0):
    print('Orbit for seed {0}, growth rate of {1}, plotting {2} iterations after skipping {3}'.format(seed, r, n_iter, n_skip))
    X=[]
    T=[]
    t=0
    x = seed
    # Iterate the logistic equation, printing only if n_skip steps have been skipped
    for i in range(n_iter + n_skip):
        if i >= n_skip:
            X.append(x)
            T.append(t)
            t+=1
        x = logistic_eq(r,x)

    return T,X

def bifurcation_diagram(seed, n_skip, n_iter, step=0.0001, r_min=0,r_max=4):
    print("Starting with x0 seed {0}, skip plotting first {1} iterations, then plot next {2} iterations.".format(seed, n_skip, n_iter))

    R = []
    X = []
    r_range = np.linspace(r_min, r_max, int(1/step))

    for r in r_range:
        x = seed
        # For each r, iterate the logistic function and collect datapoint if n_skip iterations have occurred
        for i in range(n_iter+n_skip+1):
            if i >= n_skip:
                R.append(r)
                X.append(x)

            x = logistic_eq(r,x)

    return R,X

def plot_equation_orbit(T,X):
    ax = plt.subplot(1,1,1)
    plot_equation_orbitAx(T,X,ax)
    plt.show()

def plot_equation_orbitAx(T,X,ax):
    ax.plot(T, X)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, T[-1])
    ax.set_xlabel('Time t')
    ax.set_ylabel('X')


def plotBifurcation(R,X,r_min=0, r_max=4.0):
    ax = plt.subplot(1,1,1)
    plotBifurcationAx(R,X,r_min,r_max,ax)
    plt.show()

def plotBifurcationAx(R,X,r_min=0, r_max=4.0,ax=None):
    ax.plot(R, X, ls='', marker=',')
    ax.set_ylim(0, 1)
    ax.set_xlim(r_min, r_max)
    ax.set_xlabel('r')
    ax.set_ylabel('X')

def singleBifurcation():
    rMin = 0
    rMax = 9
    R,X = bifurcation_diagram(0.2, 5, 10, r_min=rMin,r_max=rMax)
    ax = plt.subplot(1,1, 1)
    ax.set_title('LogisticMap:r*x*(1-x)**3')
    plotBifurcationAx(R,X,rMin,rMax,ax)
    plt.show()

def multBifurcation():
    rMin = 0
    rMax = 9

    nSkips=[1,2,5,10,20,50]
    plt.suptitle("LogisticMap:r*x*(1-x)**3")
    for i,nSkips in enumerate(nSkips):
        R,X = bifurcation_diagram(0.2, nSkips, 5, r_min=rMin,r_max=rMax)

        ax = plt.subplot(3, 2, i+1)
        ax.set_title('nSkips='+str(nSkips))
        plotBifurcationAx(R,X,rMin,rMax,ax)
    plt.show()

    '''
    nIters=[1,5,10,20,50,100]
    plt.suptitle("LogisticMap:r*x*(1-x)**3")
    for i,nIter in enumerate(nIters):
        R,X = bifurcation_diagram(0.2, 1, nIter, r_min=rMin,r_max=rMax)

        ax = plt.subplot(3, 2, i+1)
        ax.set_title('nIter='+str(nIter))
        plotBifurcationAx(R,X,rMin,rMax,ax)
    plt.show()
    '''

def main():
    #x:= r*x*(1-x)
    #logistic_equation_orbit(0.1, 3.05, 100)
    #logistic_equation_orbit(0.1, 3.9, 100)
    #T,X = logistic_equation_orbit(0.2, 7, 100, 1000)
    #plot_equation_orbit(T,X)
    #return

    '''
    rl = [6, 6.5, 7, 7.5, 8.5, 9]
    for i,r in enumerate(rl):
        T,X = logistic_equation_orbit(0.2, r, 100, 1000)

        ax = plt.subplot(3, 2, i+1)
        ax.set_title('r='+str(r))
        plot_equation_orbitAx(T,X,ax)
    plt.show()
    return
    '''

    #bifurcation_diagram(0.2, 100, 5)
    #bifurcation_diagram(0.2, 100, 10)
    #bifurcation_diagram(0.2, 100, 10, r_min=2.7)
    #rMin = 0
    #R,X = bifurcation_diagram(0.2, 100, 10, r_min=rMin)
    #plotBifurcation(R,X,r_min=rMin)

    #x:= r*x*(1-x)**2

    multBifurcation()


if __name__=='__main__':
    main()

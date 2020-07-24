#Python3 Steven
#common proability distrubutions
import numpy as np
import math
from scipy import integrate
from scipy.special import gamma,beta,factorial
from sklearn.preprocessing import Normalizer
#https://en.wikipedia.org/wiki/Gamma_function
#https://en.wikipedia.org/wiki/Beta_function
#https://en.wikipedia.org/wiki/Factorial
from permutationAndCom import permut,combinat


#---------------------------Discrete distribution------------------------------#
def Discrete_uniform_distribution(x,N=5):#https://en.wikipedia.org/wiki/Discrete_uniform_distribution
    return np.zeros(len(x)) + 1/N

def Binomial_distribution(N,p=1): #https://en.wikipedia.org/wiki/Binomial_distribution
    assert(N>=1 and 0<=p<=1)
    def Binomial(k):
        return combinat(N,k)*np.power(p,k)*np.power(1-p,N-k)
    
    return list(map(Binomial, [i for i in range(N)]))
    
def Geometric_distribution(N,p):#https://en.wikipedia.org/wiki/Geometric_distribution
    def Geometric(k):
        return np.power(1-p,k)*p
    
    return list(map(Geometric, [i for i in range(N)]))

def Hypergeometric_distribution(N,K,n): #https://en.wikipedia.org/wiki/Hypergeometric_distribution
    def Hypergeometric(k):
        return combinat(K,k)*combinat(N-K,n-k)/combinat(N,n)
    
    return list(map(Hypergeometric, [i for i in range(n)]))

def Poisson_distribution(N=20, lam=1): #https://en.wikipedia.org/wiki/Poisson_distribution
    def Poisson(k):
        return np.power(lam,k)*np.power(np.e, -1*lam)/factorial(k)
    
    return list(map(Poisson, [i for i in range(N)]))

def ZipfsLaw(N=10, s=1):#https://en.wikipedia.org/wiki/Zipf%27s_law
    v = np.sum(1/np.power([i+1 for i in range(N)],s))
    def ZipfLaw(k):
        return (1/np.power(k,s))/v
    
    return list(map(ZipfLaw, [i+1 for i in range(N)]))

def Beta_binomial_distribution(N=10,alpha=0.2,bta=0.25): #https://en.wikipedia.org/wiki/Beta-binomial_distribution
    def Beta_binomial(k):
        return combinat(N,k)*beta(k+alpha, N-k+bta)/beta(alpha,bta)
    
    return list(map(Beta_binomial, [i for i in range(N)]))

def Logarithmic_distribution(N,p=0.33): #https://en.wikipedia.org/wiki/Logarithmic_distribution
    def Logarithmic(k):
        return -1*np.power(p,k)/(np.log(1-p)*k)
    
    return list(map(Logarithmic, [i+1 for i in range(N)]))

def Conway_Maxwell_Poisson_distribution(N=20,lam=1, v=1.5): # https://en.wikipedia.org/wiki/Conway%E2%80%93Maxwell%E2%80%93Poisson_distribution
    def Conway(k):
        return np.power(lam,k)/np.power(factorial(k),v)
    
    max = 10000
    z = np.sum(list(map(Conway, [i for i in range(max)])))
    res = list(map(Conway, [i for i in range(N)]))
    return res/z

def Modified_Bessel_functions(x,alpha=0): #https://en.wikipedia.org/wiki/Bessel_function#Modified_Bessel_functions
    def Bessel(k):
        return 1/(factorial(k)*gamma(k+alpha+1))*np.power(0.2*x,2*k+alpha)
    
    max = 10000
    return np.sum(list(map(Bessel, [i for i in range(max)])))

def Skellam_distribution(N=8,u1=1,u2=1): #https://en.wikipedia.org/wiki/Skellam_distribution
    def Skellam(k):
        return np.exp(-u1-u2)*np.power(u1/u2, 0.5*k)*Modified_Bessel_functions(x=2*np.sqrt(u1*u2),alpha=k)
    return list(map(Skellam, [i-N/2 for i in range(N+1)]))
    
def Yule_Simon_distribution(N=20,ru=0.25): #https://en.wikipedia.org/wiki/Yule%E2%80%93Simon_distribution
    def Yule_Simon(k):
        return ru*beta(k,ru+1)
    return list(map(Yule_Simon, [i+1 for i in range(N)]))
    
def Riemann_zeta_function(s=2):#https://en.wikipedia.org/wiki/Riemann_zeta_function
    def Riemann_zeta(k):
        return 1/np.power(k,s)
    
    max = 100
    return np.sum(list(map(Riemann_zeta, [i+1 for i in range(max)])))

def Zeta_distribution(N=16,s=2): #https://en.wikipedia.org/wiki/Zeta_distribution
    def Zeta(k):
        return 1/(Riemann_zeta_function(s=s)*np.power(k,s))
    return list(map(Zeta, [i+1 for i in range(N)]))

#---------------------------Continuous distribution------------------------------#
def Uniform_distribution(N,a=1,b=3): #https://en.wikipedia.org/wiki/Uniform_distribution_(continuous)
    return np.zeros(N) + 1/(b-a)

def Error_function(x): #https://en.wikipedia.org/wiki/Error_function
    def erf(t):
            return (2/np.sqrt(np.pi))*np.exp(-t**2)
    return integrate.quad(erf, 0, x)[0]

def NormalDistribution(x, delta=1, u=0): #https://en.wikipedia.org/wiki/Normal_distribution
    return (1/(delta*np.sqrt(2*np.pi)))*np.exp(-0.5*((x-u)/delta)**2)
  
def Cauchy_pdf(x, x0=0, scaler=1):#https://en.wikipedia.org/wiki/Cauchy_distribution
    return (1/(np.pi*scaler))*(scaler**2/((x-x0)**2+scaler**2))
  
def Generalized_logistic_distribution(x,alpha=1):#https://en.wikipedia.org/wiki/Generalized_logistic_distribution
    return alpha*np.exp(-1*x)/np.power(1+np.exp(-1*x),alpha+1)

def Gumbel_distribution(x,u=0,belta=1):#https://en.wikipedia.org/wiki/Gumbel_distribution
    z = (x-u)/belta
    return np.exp(-1*(z + np.exp(-1*z)))/belta

def Laplace_distribution(x,u=0,b=1):#https://en.wikipedia.org/wiki/Laplace_distribution
    return np.exp(-1*np.abs(x-u)/b)
   
def Logistic_distribution(x, u=0, s=1):#https://en.wikipedia.org/wiki/Logistic_distribution
    z = np.exp(-1*(x-u)/s)
    return z/(s*(1+z)**2)

def Gamma_distribution(x,k=1,theta=2): #https://en.wikipedia.org/wiki/Gamma_distribution
    return np.power(x,k-1)*np.exp(-1*x/theta)/(gamma(k)*np.power(theta,k))

def StudentT_distribution(x,v=1):#https://en.wikipedia.org/wiki/Student%27s_t-distribution
    z = np.power(1+x**2/v, -0.5*(v+1))
    return z*gamma((v+1)/2)/(gamma(v/2)*np.sqrt(np.pi*v))
    #return z/(beta(0.5, v/2)*np.sqrt(v))

def Log_normal_distribution(x, delta=1, u=0):#https://en.wikipedia.org/wiki/Log-normal_distribution
    z = -1*(np.log(x)-u)**2/(2*delta**2)
    return (1/(x*delta*np.sqrt(2*np.pi)))* np.exp(z)

def Weibull_distribution(x,lamda=1,k=1): #https://en.wikipedia.org/wiki/Weibull_distribution
    y = np.zeros((len(x),))
    l = np.where(x < 0)
    if len(l) != 0:
        y[l[0]]=0
        
    l = np.where(x >= 0)
    if len(l) != 0:
        y[l[0]] = (k/lamda)*np.power(x[l[0]]/lamda,k-1)*np.exp(-1*np.power(x[l[0]]/lamda,k))
    return y

def Pareto_distribution(x,alpha=1,Xm=1):#https://en.wikipedia.org/wiki/Pareto_distribution
    y = np.zeros((len(x),))
    l = np.where(x < Xm)
    if len(l) != 0:
        y[l[0]]=0
    
    l = np.where(x >= Xm)
    if len(l) != 0:
        y[l[0]] = alpha*np.power(Xm,alpha)/np.power(x[l[0]],alpha+1)
    return y

def Rayleigh_distribution(x,delta=0.5):#https://en.wikipedia.org/wiki/Rayleigh_distribution
    y = np.zeros((len(x),))
    l = np.where(x >= 0)
    if len(l) != 0:
        y[l[0]] = (x/delta**2)*np.exp(-1*x**2/(2*delta**2))
    return y
    
def Beta_distribution(x,alpha=0.5,bta=0.5): #https://en.wikipedia.org/wiki/Beta_distribution
    z = gamma(alpha)*gamma(bta)/gamma(alpha+bta)
    return np.power(x, alpha-1)*np.power(1-x, bta-1)/z
 
def Chi_distribution(x,k=1): #https://en.wikipedia.org/wiki/Chi_distribution
    return (2/ (np.power(2,k/2)* gamma(k/2)))*np.power(x,k-1)*np.exp(-0.5*x**2)

def Erlang_distribution(x,k=1,u=2.0): #https://en.wikipedia.org/wiki/Erlang_distribution
    return np.power(x,k-1)*np.exp(-1*x/u)/(np.power(u,k)*factorial(k-1))

def Exponential_distribution(x,lam=1): #https://en.wikipedia.org/wiki/Exponential_distribution
    return lam*np.exp(-1*lam*x)

def Von_Mises_distribution(x,u=0,k=0): #https://en.wikipedia.org/wiki/Von_Mises_distribution
    return np.exp(k*np.cos(x-u))/(2*np.pi*Modified_Bessel_functions(alpha=0,x=k))

def Boltzmann_distribution(x,a=1): #https://en.wikipedia.org/wiki/Maxwell%E2%80%93Boltzmann_distribution
    return x**2*np.sqrt(2/np.pi)*np.exp(-0.5*x**2/a**2)/a**3

def logit(x):#https://en.wikipedia.org/wiki/Logit
    return np.log(x/(1-x))

def Logit_normal_distribution(x,deta=0.2,u=0): #https://en.wikipedia.org/wiki/Logit-normal_distribution
    z = np.exp(-1*((logit(x)-u)**2)/(2*deta**2))
    return z/(deta*np.sqrt(2*np.pi)*x*(1-x))

def sgn(x,k):
    if x<k:
        return -1
    elif x==k:
        return 0
    return 1
    
def Irwin_Hall_distribution(x,n=1): #https://en.wikipedia.org/wiki/Irwin%E2%80%93Hall_distribution
    def Irwin_Hall(x):
        def f(k):
            return np.power(-1,k)*combinat(n,k)*np.power(x-k,n-1)
        #print('x_=',np.floor(x))
        NN = int(np.floor(x))
        res = np.array(list(map(f, [i for i in range(NN+1)])))
        res = np.sum(res)*(1/factorial(n-1))
        return res
    
    return list(map(Irwin_Hall, [i for i in x]))
    
def Bates_distribution(x,n=1,a=0,b=1): #https://en.wikipedia.org/wiki/Bates_distribution
    def Bates(x):
        def f(k):
            return np.power(-1,k)*combinat(n,k)*np.power((x-a)/(b-a)-k/n,n-1)*sgn((x-a)/(b-a),k/n)
        return np.sum(list(map(f, [i for i in range(n)])))
    
    return list(map(Bates, [i for i in x]))

def Kumaraswamy_distribution(x,a=0.5,b=0.5): #https://en.wikipedia.org/wiki/Kumaraswamy_distribution
    return a*b*np.power(x,a-1)*np.power(1-np.power(x,a),b-1)

def PERT_distribution(x,a=0,b=10,c=100): #https://en.wikipedia.org/wiki/PERT_distribution
    alpha = 1+4*(b-a)/(c-a)
    bta = 1+4*(c-b)/(c-a)
    return np.power(x-a, alpha-1)*np.power(c-x, bta-1)/(beta(alpha, bta)*np.power(c-a, alpha+bta-1))

def Reciprocal_distribution(x,a=1,b=4): #https://en.wikipedia.org/wiki/Reciprocal_distribution
    return 1/x*np.log(b/a)

def Triangular_distribution(x,a=1,c=3,b=4): #https://en.wikipedia.org/wiki/Triangular_distribution
    assert(a<c and c<b)
    y = np.zeros((len(x),))
    l = np.where(x > b)
    if len(l) != 0:
        y[l[0]]=0
        
    l = np.where(x <= b)
    if len(l) != 0:
        y[l[0]]= 2*(b-x[l[0]])/((b-a)*(b-c))
    l = np.where(x <= c)
    if len(l) != 0:
        y[l[0]]= 2*(x[l[0]]-a)/((b-a)*(c-a))
    l = np.where(x < a )
    if len(l) != 0:
        y[l[0]]= 0
    return y    
        
def Trapezoidal_distribution(x,a=-3,b=-2,c=-1,d=0): #https://en.wikipedia.org/wiki/Trapezoidal_distribution
    assert(a<b and b<c and c<d)
    y = np.zeros((len(x),))
    l = np.where(x <= d)
    if len(l) != 0:
        y[l[0]]=2*(d-x[l[0]])/((d+c-a-b)*(d-c))
        
    l = np.where(x <= c)
    if len(l) != 0:
        y[l[0]]=2/(d+c-a-b)
    
    l = np.where(x < b)
    if len(l) != 0:
        y[l[0]]=2*(x[l[0]]-a)/((d+c-a-b)*(b-a))
    return y

def PIFuc(x):
        return 0.5*(1+Error_function(x/np.sqrt(2)))
        
def Truncated_normal_distribution(x,a=-10,b=10,u=-8,deta=2): #https://en.wikipedia.org/wiki/Truncated_normal_distribution
    return (1/deta)*NormalDistribution((x-u)/deta)/(PIFuc((b-u)/deta)-PIFuc((a-u)/deta))

def Wigner_semicircle_distribution(x,r=0.25): #https://en.wikipedia.org/wiki/Wigner_semicircle_distribution
    return 2*np.sqrt(r**2-x**2)/(np.pi*r**2)

def Dagum_distribution(x,b=1,p=1,a=0.5): #https://en.wikipedia.org/wiki/Dagum_distribution
    z1=(a*p/x)*np.power(x/b, a*p)
    z2 = np.power(np.power(x/b, a)+1, p+1)
    return z1/z2

def F_distribution(x,d1=1,d2=1): #https://en.wikipedia.org/wiki/F-distribution
    z = np.power(d1*x,d1)*np.power(d2,d2)/np.power(x*d1+d2,d1+d2)
    return np.sqrt(z)/(x*beta(0.5*d1,0.5*d2))

def Folded_normal_distribution(x,u=1,deta=1): #https://en.wikipedia.org/wiki/Folded_normal_distribution
    z=1/(deta*np.sqrt(2*np.pi))
    z1 = z*np.exp(-0.5*(x-u)**2/deta**2)
    z2 = z*np.exp(-0.5*(x+u)**2/deta**2)

    y = z1+z2
    l = np.where(x < 0)
    if len(l) != 0:
        y[l[0]]=0
    return y    

def Frechet_distribution(x,alpha=1,s=1,m=0): #https://en.wikipedia.org/wiki/Fr%C3%A9chet_distribution
    z=(x-m)/s
    z1=np.power(z, -1-alpha)
    z2 = np.exp(-1*np.power(z, -1*alpha))
    return (alpha/s)*z1*z2

def Gompertz_distribution(x,eta=0.1,b=1): #https://en.wikipedia.org/wiki/Gompertz_distribution
    z=eta+b*x-eta*np.exp(b*x)
    return b*eta*np.exp(z)

def Half_normal_distribution(x,delta=1): #https://en.wikipedia.org/wiki/Half-normal_distribution
    return np.sqrt(2)/(delta*np.sqrt(np.pi))*np.exp(-0.5*x**2/delta**2)

def Nakagami_distribution(x,m=0.5,w=1): #https://en.wikipedia.org/wiki/Nakagami_distribution
    z = gamma(m)*np.power(w,m)
    return (2*np.power(m,m)/z)*np.power(x,2*m-1)*np.exp(-1*m*x**2/w)

def Levy_distribution(x,u=0,c=0.5): #https://en.wikipedia.org/wiki/L%C3%A9vy_distribution
    return np.sqrt(0.5*c/np.pi)*np.exp(-0.5*c/(x-u))/np.power(x-u, 1.5)

def Shifted_Gompertz_distribution(x,b=0.4,ni=0.01): #https://en.wikipedia.org/wiki/Shifted_Gompertz_distribution
    z=np.exp(-1*b*x)
    return b*z*np.exp(-1*ni*z)*(1+ni*(1-z))

def JohnsonSU_distribution(x,gama=-2,deta=2,xi=1.1,lam=1.5): #https://en.wikipedia.org/wiki/Johnson%27s_SU-distribution
    z1=deta/(lam*np.sqrt(2*np.pi))
    z2=1/np.sqrt(1+((x-xi)/lam)**2)
    z3=np.exp(-0.2*(gama+deta*np.arcsinh((x-xi)/lam))**2)
    return z1*z2*z3

def Landau_distribution(x,u=0,c=1): #https://en.wikipedia.org/wiki/Landau_distribution
    def getFunctionSympy(t,x=x,u=u,c=c):
            #print('x,u,c=',x,u,c)
            y=(1/np.pi*c)*np.exp(-t)*np.cos(t*((x-u)/c)+ 2*t/np.pi*np.log(t/c))
            #y=np.exp(-t)
            return y
    
    return integrate.quad(getFunctionSympy, 0, np.inf)[0]

def Stable_distribution(x,beta=0,c=1,alpha=0.5): #https://en.wikipedia.org/wiki/Stable_distribution
    def sgnArray(x,k):
        return np.where(x>k, 1, 0) + np.where(x<k, -1, 0)
         
    z1 = np.power(np.abs(x),1+alpha)
    z2 = np.power(c,alpha)*(1+sgnArray(x,0)*beta)*np.sinh(0.5*np.pi*alpha)
    z3 = gamma(alpha+1)/np.pi
    res = z2*z3/z1
    # res = res.reshape(1,-1)
    # res = Normalizer().fit(res).transform(res)
    # res = res.reshape((len(x)))
    # print(res.shape,len(x))
    return res

def PIFuc2(x):
    #return 0.5*(1+Error_function(x/np.sqrt(2)))
    return np.array(list(map(PIFuc, [i for i in x])))
    
def Skew_normal_distribution(x,alpha=-4): #https://en.wikipedia.org/wiki/Skew_normal_distribution
    return 2* NormalDistribution(x)*PIFuc2(alpha*x)

def Noncentral_t_distribution(x,u=0,v=1): #https://en.wikipedia.org/wiki/Noncentral_t-distribution
    return StudentT_distribution(x,v=v)



#------------------------------------main----------------------------------#   
def main():
    pass 
    
if __name__ == '__main__':
    main()

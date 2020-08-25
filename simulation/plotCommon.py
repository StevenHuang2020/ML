import matplotlib.pyplot as plt

def plot(x,y):
    plt.plot(x,y)
    plt.show()

# def plot(x,y=None):
#     if y:
#         plt.plot(x,y)
#     else:
#         plt.plot(x)
#     plt.show()
    
def plotSub(x,y,ax=None, aspect=False, legend=False, label='',linestyle='solid',marker=''):
    ax.plot(x,y,label=label,marker=marker,linestyle=linestyle)
    #ax.title.set_text(name)
    if aspect:
        ax.set_aspect(1)
    if legend:
        ax.legend()

def scatterSub(x,y,ax=None,label='',marker='.'):
    ax.scatter(x,y,linewidths=.3, label=label, marker=marker)
    ax.legend()
    


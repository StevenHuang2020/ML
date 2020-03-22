# Python3 steven
#https://en.wikipedia.org/wiki/Mandelbrot_set
import numpy as np
import matplotlib.pyplot as plt

def genMandelBrot(N=1000):
    M = np.zeros([N, N], int)
    xvalues = np.linspace(-2, 2, N)
    yvalues = np.linspace(-2, 2, N)

    for u, x in enumerate(xvalues):
        for v, y in enumerate(yvalues):
            z = 0 + 0j
            c = complex(x, y)
            for i in range(100):
                z = z * z + c
                if abs(z) > 2.0:
                    M[v, u] = 1
                    break
    return M

def main():
    plt.imshow(genMandelBrot())
    plt.show()

if __name__=='__main__':
    main()

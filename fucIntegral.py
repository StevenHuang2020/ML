#steven 02/03/2020
#function integral to calculate area of curve and x axis

def fun(x):
    #return x 
    return x**2

def main():
    """calculate area y=x^2  form x = 0 to 1
    divide 0~1 to N times,  erevy part 
    """
    N = 1000

    s = 0
    for i in range(N):
        #print(i)
        x = 1/N
        y = fun(i/N)
        s += x*y
    
    print(s)

if __name__ == '__main__':
    main()

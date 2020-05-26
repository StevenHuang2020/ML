#Python3 Steven
#permutation and combination
import numpy as np
from itertools import combinations, permutations

def permut(N, m): #A{N,m}
    return len(list(permutations(range(N),m)))

    # l = len(array)
    # p = list(permutations(range(l)))
    # #p = np.array()
    # # print('p=',p)
    # # print(array[p,:])
    # print('A_[{},{}]={}'.format(l, N, array[p, :]))
    
def combinat(N, m):
    return len(list(combinations(range(N),m)))
    #return combinations(range(N),m)

def main():    
    print('A_{4,3}=',list(permutations(range(4),3)))
    print('C_{4,3}=',list(combinations(range(4),3)))
    print('A_{3,2}=',list(permutations([1, 2, 3], 2)))
    print('C_{3,2}=',list(combinations([1, 2, 3], 2)))

    #A = np.array([[1, 2, 4]]) # np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    #permut(A, 2)
    
    print(permut(5,2))
    print(combinat(5,2))
    
    N=25
    for i in range(N):
        print('N,i,combinat=',N,i,combinat(N,i))
    
if __name__=='__main__':
    main()
    
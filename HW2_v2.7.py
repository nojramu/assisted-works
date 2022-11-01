# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 20:49:13 2022

@author: Leah
"""

from scipy.sparse import diags
import scipy.linalg as linalg 
import numpy as np

def array_a():
    global a
    #source: https://stackoverflow.com/questions/5842903/block-tridiagonal-matrix-python
    n = 100
    k = [-1*np.ones(n-2),np.ones(n-1),4*np.ones(n),np.ones(n-1),-1*np.ones(n-2)]
    offset = [-2,-1,0,1,2]
    a = diags(k,offset).toarray()
    #print("Array_a",a)
    #print(type(a))


def array_b():
    global b
    #source: https://stackoverflow.com/questions/7154739/how-can-i-get-an-array-of-alternating-values-in-python
    b = np.empty((100,))
    b[::2] = 0.999
    b[1::2] = 1.001
    #print("Array_b\n",b)
    #print(type(b))


def small_array():
    global a
    global b
    a = np.array([[-5, -1, 2],
                  [2, 6, -3],
                  [2, 1, 7]])

    b = np.array([1, 2, 32])


def Methods(z):
    if z == 0:#LU Decomposition
        #source: https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.lu_solve.html
        lu, piv = linalg.lu_factor(a)
        x = linalg.lu_solve((lu,piv), b)
        print("LU Decomposition\n",x)

    elif z==1:#Matrix Inversion
        #source: https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.inv.html
        x = linalg.inv(a)
        #x = np.dot(a, linalg.inv(a))
        print("Matrix Inversion\n",x)

    elif z==2:#Cramer's rule
        #source: https://stackoverflow.com/questions/70323836/solving-system-of-linear-equation-using-cramers-method-in-python
        n = len(b)
        mask = np.broadcast_to(np.diag(np.ones(n)), [n, n, n]).swapaxes(0, 1)
        Ms = np.where(mask, np.repeat(b, n).reshape(n, n), a)
        x = np.linalg.det(Ms) / np.linalg.det(a)
        print("Cramer's rule\n",x)

    elif z==3:#Gauss-Seidel
        #source: https://www.geeksforgeeks.org/gauss-seidel-method
        n = len(a)
        iterations = 7
        x = np.zeros(len(b))
        print("Gauss-Seidel")
        for it_count in range(0, iterations+1):
            print("iteration {0}: {1}".format(it_count,x))
            for j in range(0, n):        
                d = b[j]                  
                for i in range(0, n):     
                    if(j != i):
                        d-=a[j][i] * x[i]     
                x[j] = d / a[j][j]

    else:#Successive Over Relaxation
        #source: https://stackoverflow.com/questions/53251299/successive-over-relaxation
        iterations = 100
        x = np.zeros(len(b))
        print("Successive Over Relaxation")
        for it_count in range(0, iterations+1):
            x_new = np.zeros_like(x)
            print("iteration {0}: {1}".format(it_count,x))
            for i in range(a.shape[0]):
                s1 = np.dot(a[i, :i], x_new[:i])
                s2 = np.dot(a[i, i + 1:], x[i + 1:])
                x_new[i] = (b[i] - s1 - s2) / a[i, i]
            if np.allclose(x, x_new, rtol=1e-8):
                break
            x = x_new


for z in range(5):
    #array_a()
    #array_b()
    small_array()
    #print("a =", a,"\nb =", b)
    Methods(z)

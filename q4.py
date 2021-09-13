import pandas as pd
import numpy as np
import  matplotlib.pyplot as plt
import math
def numV(x):
    if(x=='Alaska'):
        return 0
    else:
        return 1
def muV(a,mu0,mu1):
    if(a==0):
        return mu0
    else:
        return  mu1
def main():
    #load the data
    fileX="./data/q4/q4x.dat"
    fileY="./data/q4/q4y.dat"
    x=pd.read_csv(fileX, delimiter=" ", header=None)
    x=x.dropna(axis=1)
    #x1 x2
    x=np.array(x)

    #normalize the data
    mean=np.average(x,axis=0)
    std=np.std(x,axis=0)
    x=(x-mean)/std
    #x0 x1 x2
    y=pd.read_csv(fileY, header=None)
    y=np.array(y).reshape(-1,1)
    yA=y
    y=np.array([numV(x) for x in y]).reshape(-1,1)
    phi=np.sum(y)/(y.shape[0])
    mu0=np.sum((1-y)*x, axis=0)
    mu0/=(np.sum(1-y))
    mu1=np.sum((y)*x, axis=0)
    mu1/=(np.sum(y))

    muY=([muV(a,mu0.tolist(),mu1.tolist()) for a in y])
    muY=np.array(muY)
    #since x is m*n matrix,we take x-muY traspose and do matrix multipliation with x-muY to get dimension n*n. The resulting vector already has the summation of m values. to get sigma we just divide by m
    sigma=((x-muY).T)@(x-muY)
    sigma/=x.shape[0]
    # print(mu0)
    # print(mu1)
    # print(sigma)
    mu0=mu0.reshape((1,-1))
    mu1=mu1.reshape((1,-1))
    sigma0=(1-y)*(x-mu0)
    #sigma0 now has all the values that have y=0. matmul this with x-mu0 to get matrix of size n*n. This would have the summation already of relevant values(all examples with y=0)
    sigma0=((x-mu0).T)@(sigma0)
    sigma0/=np.sum(1-y)
    sigma1=(y)*(x-mu1)
    sigma1=((x-mu1).T)@(sigma1)
    sigma1/=np.sum(y)
    # print(sigma0)
    # print(sigma1)

    #CALCULATE AND PLOT THE LINEAR BOUNDARY
    #in my defination mu1 is 1*n matrix and not n*1
    c=0.5*(mu1@(np.linalg.inv(sigma))@(mu1.T) -(mu0@(np.linalg.inv(sigma))@(mu0.T))) +np.log((1-phi)/phi)
    mat=(mu1-mu0)@(np.linalg.inv(sigma))
    #now to plot I have only looked at the first two values of mat
    a=mat[0][0]
    b=mat[0][1]
    #the equation of line is: ax1+bx2=c
    plt.scatter(x.T[0],x.T[1], c=y)
    x1 = np.linspace(-2,3,100)
    x1=np.array(x1).reshape(-1,1)
    x2=(c-a*x1)/(1*b)
    plt.plot(x1,x2, color='red')

    #CALCULATE AND PLOT THE QUADRATIC BOUNDARY

if __name__ == '__main__':
    main()

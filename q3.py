import pandas as pd
import numpy as np
import math
#load the data
fileX="./data/q3/logisticX.csv"
fileY="./data/q3/logisticY.csv"
x=pd.read_csv(fileX, delimiter=",", header=None)
#x1 x2
x=np.array(x)
#normalize the data
mean=np.average(x,axis=0)
std=np.std(x,axis=0)
x=(x-mean)/std
#x0 x1 x2
x=np.append(np.ones((x.shape[0],1)), x, axis=1)
y=pd.read_csv(fileY, header=None)
y=np.array(y)

theta=np.array([0, 0, 0])
theta=np.reshape(theta,(-1,1))
corpus_size=x.shape[0]
converged=False
leaning_rate=0.01
llTheta=0
pLLTheta=0
epsilon=10e-6
firstPass=True
epochs=0

#dimension of x: m*n; mis number of examples, n is number of dimensions
#dimension of theta: n*1
#dimension of y: m*1
while not converged:
    epochs+=1
    #@ means matrix multiplication
    z=math.e**(-1*(x@theta)).reshape(-1,1)
    # print(z.shape)
    h=1/(1+z)
    # print(y.shape)
    # print(h.shape)
    llThetaA=y*np.log(h)+(1-y)*np.log(1-h)
    llTheta=np.average(llThetaA, axis=0)/2
    #calculate deltheta
    deltheta=(y-h)*x
    deltheta=np.average(deltheta,axis=0).reshape(-1,1)
    deltheta/=2#formula is= (1/2m)*summation(....)
    #calculate the hessian now
    del2theta=(h**2)*z
    #now the dim of del2theta is m*1
    del2theta=del2theta*x
    #now the dim of del2theta is m*n
    hessian=[]
    for coln in del2theta.T:
        coln=coln.reshape(-1,1)
        hessian.append(coln*x)
    hessian=np.array(hessian)
    # print(hessian.shape)
    #hessian has now dimension n*m*n. For a particular row i, hessian[k][i][j] represents z*h^2 multiplied by xj and xk.
    #take an average over all examples
    hessian=np.average(hessian,axis=1)
    hessian*=-0.5# hessian has -1/2*summation(...)
    hessian=np.linalg.inv(hessian)
    # print(hessian.shape)
    # print(deltheta.shape)
    theta=theta-hessian@deltheta
    # print(hessian)
    # print(llTheta)
    # print(y.shape)
    # print(deltheta)
    # break
    if firstPass:
        firstPass=False
        pLLTheta=llTheta
        continue;
    if abs(pLLTheta-llTheta)< epsilon:
        converged=True
    pLLTheta=llTheta
print(theta)
print(epochs)

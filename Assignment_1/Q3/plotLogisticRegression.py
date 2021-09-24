def col(a):
    if a==1:
        return 'red'
    else:
        return 'blue'
import pandas as pd
import numpy as np
import  matplotlib.pyplot as plt
fileX="../../data/q3/logisticX.csv"
fileY="../../data/q3/logisticY.csv"
x=pd.read_csv(fileX, delimiter=",", header=None)
#x1 x2
x=np.array(x)
#normalize the data
mean=np.average(x,axis=0)
std=np.std(x,axis=0)
x=(x-mean)/std
#read the labels
y=pd.read_csv(fileY, header=None)
y=np.array(y)
theta=[ 0.40125316,  2.5885477 , -2.72558849]
# m=[ mark(a) for a in y]
# yc=col(y)
yc=[col(a) for a in y]
plt.scatter(x.T[0],x.T[1], c=yc)
# plt.legend(["0","1"])
# plt.scatter(x,y, label='define label')
x1 = np.linspace(-2,3,100)
x2 = (theta[1]*x1+theta[0])*(-1/theta[2])
plt.plot(x1,x2, color='green')
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Logistic Regression: Red(label 1) and Blue(label 0)")
plt.show()

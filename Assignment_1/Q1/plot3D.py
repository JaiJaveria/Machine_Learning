import numpy as np
import pandas as pd
import  matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
xFileN="./data/q1/linearXNormalized.csv"
yFileN="./data/q1/linearY.csv"
x=pd.read_csv(xFileN, delimiter=",", header=None)
x=np.append(np.ones((x.shape[0],1)), x, axis=1)
x=np.array(x)
y=pd.read_csv(yFileN, delimiter=",", header=None)
y=np.array(y)
def lossFunc(a,b):
    theta=np.array([a,b]).reshape(2,1)
    z=y-x@theta
    z**=2
    return np.average(z)/2
theta=pd.read_csv('data/q1/thetaParams.csv', delimiter=" ", header=None)
theta=np.array(theta).T
t0=0.9956
t1=0.001339
x0 = np.arange(t0-1,t0+1, 2/100)
x1 = np.arange(t1-1,t1+ 1, 2/100)
# Creating 2-D grid of features [X0, X1]
[X0,X1] = np.meshgrid(x0, x1)
Z=[[0.0 for i in range(len(x0))] for _ in range(len(x1))]
Z=np.array(Z)
for i in range(len(x0)):
    for j in range(len(x1)):
        Z[i][j]=lossFunc(X0[i][j], X1[i][j])
cost =np.array([lossFunc(theta[0][j], theta[1][j]) for j in range(theta.shape[1])])
cost=cost.reshape(-1,)

fig=plt.figure()
ax = plt.axes(projection ='3d')
ax.plot_surface(X0, X1, Z)
plt.title("Loss Function")
plt.xlabel("theta0")
plt.ylabel("theta1")
ax.set_zlabel("Loss")
for i in range(theta.shape[1]):
    ax.plot(theta[0][i],theta[1][i],cost[i],'ro')
    plt.pause(1e-30)
plt.show()

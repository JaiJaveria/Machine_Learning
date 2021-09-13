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
theta=pd.read_csv('data/q1/thetaParams.csv', delimiter=" ", header=None)
theta=np.array(theta)
t0=0.9956
t1=0.001339
x0 = np.arange(t0-2,t0+2, 4/100)
x1 = np.arange(t1-2,t1+ 2, 4/100)
# Creating 2-D grid of features [X0, X1]
[X0,X1] = np.meshgrid(x0, x1)
fig, ax = plt.subplots(1, 1)
Z=[[0.0 for i in range(len(x0))] for _ in range(len(x1))]
for i in range(len(x0)):
    for j in range(len(x1)):
        z=y-x@(np.array([X0[i][j], X1[i][j]]).reshape(2,1))
        z**=2
        Z[i][j]=np.average(z)
        Z[i][j]/=2

# plots contour lines
ax.contour(X0, X1, Z, 40)
plt.plot(theta[0],theta[1])
# ax = plt.axes(projection ='3d')
# ax.plot_surface(x0,x1,Z)
plt.show()

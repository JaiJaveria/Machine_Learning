import numpy as np
import pandas as pd
import  matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib.animation  import FuncAnimation
xFileN="./data/q1/linearXNormalized.csv"
yFileN="./data/q1/linearY.csv"
x=pd.read_csv(xFileN, delimiter=",", header=None)
x=np.append(np.ones((x.shape[0],1)), x, axis=1)
x=np.array(x)
y=pd.read_csv(yFileN, delimiter=",", header=None)
y=np.array(y)
thetaParams='data/q1/theta001.csv'
theta=pd.read_csv(thetaParams, delimiter=" ", header=None)
theta=np.array(theta).T
t0=0.9956
t1=0.001339
x0 = np.arange(t0-1,t0+1, 2/100)
x1 = np.arange(t1-1,t1+ 1, 2/100)
# Creating 2-D grid of features [X0, X1]
[X0,X1] = np.meshgrid(x0, x1)
Z=[[0.0 for i in range(len(x0))] for _ in range(len(x1))]
for i in range(len(x0)):
    for j in range(len(x1)):
        z=y-x@(np.array([X0[i][j], X1[i][j]]).reshape(2,1))
        z**=2
        Z[i][j]=np.average(z)
        Z[i][j]/=2

fig, ax = plt.subplots(1, 1)
line, = ax.plot(theta[0], theta[1])

def animate(i, line):
    line.set_data(theta[0][:i],theta[1][:i])
    return line
# plots 40  contour lines
ax.contour(X0, X1, Z, 40)
anim = FuncAnimation(fig, animate, frames = theta.shape[1], fargs=[line], interval = 2, blit = False)
plt.title("Contour plot of  Loss Function")
plt.xlabel("theta0")
plt.ylabel("theta1")
# anim.save('contourVid.gif')
# anim.imshow()
# plt.plot(theta[0],theta[1],'red')
# ax = plt.axes(projection ='3d')
# ax.plot_surface(x0,x1,Z)
plt.show()

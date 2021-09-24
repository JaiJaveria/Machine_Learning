import numpy as np
import pandas as pd
import  matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
thetaParams='data/q2/theta1e6.csv'
theta=pd.read_csv(thetaParams, delimiter=" ", header=None)
theta=np.array(theta).T
# print(theta)
fig=plt.figure()
ax = plt.axes(projection ='3d')
plt.title("Path of Theta Values")
plt.xlabel("theta0")
plt.ylabel("theta1")
ax.set_zlabel("theta2")
ax.plot(theta[0],theta[1],theta[2])
ax.set_xlim(0,3)
ax.set_zlim(0,2)
# print(theta[0].shape)
# for i in range(theta.shape[1]):
    # ax.plot(theta[0][i],theta[1][i],theta[2][i],'ro')
    # plt.pause(1e-30)
plt.show()

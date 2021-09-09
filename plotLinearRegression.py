import numpy as np
import  matplotlib.pyplot as plt
xFileN="./data/q1/linearXNormalized.csv"
yFileN="./data/q1/linearY.csv"
fileX=open(xFileN)
fileY=open(yFileN)
#read the lines from the input file
x=fileX.readlines()
y=fileY.readlines()
valX=[]
valY=[]
for v in x:
    v=v.strip('\n')
    valX.append(float(v));
for v in y:
    v=v.strip('\n')
    valY.append(float(v));
fileX.close()
fileY.close()
# axes = plt.gca()
# axes.set_xlim([xmin,xmax])
# axes.set_ylim([0.99,1])
valX=np.array(valX)
valY=np.array(valY)
plt.scatter(valX,valY)
theta0=0.9963116206491764
theta1=0.0013397811936466525
x = np.linspace(-2,5,100)
y = theta1*x+theta0
plt.plot(x,y, color='red')
plt.show()

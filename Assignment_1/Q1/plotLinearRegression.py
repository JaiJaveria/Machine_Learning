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
valX=np.array(valX)
valY=np.array(valY)
plt.scatter(valX,valY)
plt.title('Linear Regression on Density and Acidity of Wine')
plt.xlabel('Acidity of Wine (Normalized)')
plt.ylabel('Density of Wine')
theta0=0.995640206854871
theta1= 0.0013388783159163085
x = np.linspace(-2,5,100)
y = theta1*x+theta0
plt.plot(x,y, color='red')
plt.show()

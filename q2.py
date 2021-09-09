import pandas as pd
import numpy as np
#to perform batch gradient descent for optimizing J(theta) according to question 1a.
# TODO: update the file to take inputs of files.

fileN="./data/q2/sampledPointsSmall.csv"
data=pd.read_csv(fileN, delimiter=" ")
#x1 x2 y
#0 1 2
data=np.array(data)
x=data[:,[0,1]]
#x0 x1 x2
x=np.append(np.ones((x.shape[0],1)), x, axis=1)
y=data[:,[2]]
theta=np.array([0,0,0])

dataSize=x.shape[0]
# print(dataSize)
batch_size=100
converged=False
leaning_rate=0.001
epsilon=10e-2
pAvgJT=-1;
firstPass=True
#convergence criteria. I run the training for the whole epoch and compute the average JTheta that I got. and then compare it with the previous average and stop when that value is less than epsilon.
epochs=0
while not converged:
    epochs+=1
    singlePass=False
    avgJT=0#average jTheta
    batch_num=0
    row=0
    while row<dataSize:
        batch_num+=1
        xb=x[row:row+batch_size,]
        yb=y[row:row+batch_size,]
        row+=batch_size
        #calculate loss function
        jt=yb-np.reshape(np.sum(xb*theta, axis=1), (-1,1) )
        jTheta=jt**2;
        jTheta=np.average(jTheta)/2
        #calculate gradient
        grad=jt*xb
        grad=np.average(grad, axis=0)
        #update
        theta=theta+ leaning_rate*grad;
        avgJT+=jTheta;
    avgJT/=batch_num
    if firstPass:
        firstPass=False
        pAvgJT=avgJT
        continue;
    if abs(pAvgJT-avgJT)< epsilon:
        converged=True
    pAvgJT=avgJT


print(theta)
print("epochs ",epochs)

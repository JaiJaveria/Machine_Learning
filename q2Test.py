import numpy as np
import pandas as pd
datafile='./data/q2/q2test.csv'
data=pd.read_csv(datafile)
data=np.array(data)
y=data[:,[2]]
x=data[:,[0,1]]
x=np.append(np.ones((x.shape[0],1)), x, axis=1)
# print(data)
# print(x)
# print(y)
batch_size=[1,100,1e4,1e6,00]
theta=[[2.99535658, 0.99836632, 1.9015134 ], [2.99968856, 0.99626985, 1.99723052], [2.65544195, 1.07550397, 1.97548108] , [0.64751335, 1.45492499, 1.65080709], [3,1,2]]
theta=[np.array(x) for x in theta]
print(theta)
for i in range(len(theta)):
    print(batch_size[i])
    jt=y-np.reshape(np.sum(x*theta[i], axis=1), (-1,1))
    jt**=2
    print(np.average(jt)/2)

import numpy as np
s=int(1e6)
x1=np.random.normal(loc=3, scale=2, size=s )
x2=np.random.normal(loc=-1, scale=2, size=s)
e=np.random.normal(loc=0, scale=2**0.5, size=s)
theta=[3,1,2]
# y=[]
print("x1 x2 y")
for i in range(s):
    print(x1[i],x2[i], theta[0]+theta[1]*x1[i]+ theta[2]*x2[i]+e[i])

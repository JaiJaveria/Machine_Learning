#to perform batch gradient descent for optimizing J(theta) according to question 1a.
# TODO: update the file to take inputs of files.
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
corpus_size=100
theta1=0
theta0=0
converged=False
leaning_rate=0.1
jTheta=0
epsilon=1e-8
m=len(valX)
# print(m)
i=0
firstPass=True
epochs=0
pJTheta=0
while not converged:
    epochs+=1
    sum1=0
    sum0=0
    jTheta=0
    # jt=0
    for i in range(corpus_size):
        sum1+=(valY[i]-theta1*valX[i]-theta0)*valX[i];
        sum0+=(valY[i]-theta1*valX[i]-theta0);
        jTheta+=(valY[i]-theta1*valX[i]-theta0)**2
    sum0/=(corpus_size)
    sum1/=(corpus_size)
    jTheta/=2*(corpus_size);
    theta0=theta0+ leaning_rate*sum0;
    theta1=theta1+ leaning_rate*sum1;
    print(theta0,theta1)
    if firstPass:
        firstPass=False
        pJTheta=jTheta
        continue;
    if abs(pJTheta-jTheta)< epsilon:
        converged=True
    pJTheta=jTheta

# print("theta0 ",theta0)
# print("theta1 ",theta1)
# print("epochs ",epochs)

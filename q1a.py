#to perform batch gradient descent for optimizing J(theta) according to question 1a.
# TODO: update the file to take inputs of files.
xFileN="./data/q1/linearXNormalized.csv"
yFileN="./data/q1/linearYNormalized.csv"
fileX=open(xFileN)
fileY=open(yFileN)
#read the lines from the input file
x=fileX.readlines()
y=fileY.readlines()
# print(y)
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
batch_size=100
theta=0
converged=False
leaning_rate=0.01
jTheta=0
epsilon=0.01
m=len(valX)
print(m)
i=0
pAvgJT=-1;
firstPass=True
# for _  in range(0,8000):
epochs=0
while not converged:
    epochs+=1
    singlePass=False
    avgJT=0#average jTheta
    batch_num=0
    while not singlePass:
        batch_num+=1
        sum=0
        jTheta=0
        for b in range(0,batch_size):
            if i>=m:
                singlePass=True
                i=0
                break
            else:
                sum+=(valY[i]-theta*valX[i])*valX[i];
                jTheta+=(valY[i]-theta*valX[i])**2
                i+=1;
        if i>=m:
            singlePass=True
            i=0
        # print("i: ",i)
        # print("b: ",b)
        sum/=(b+1)
        jTheta/=2*(b+1);
        avgJT+=jTheta;
        # print("s: ",sum)
        theta=theta+ leaning_rate*sum;
        # print("t: ",theta)
        # print("j: ",jTheta)
    avgJT/=batch_num
    # print("aJT: ", avgJT)
    # print("paJT: ", pAvgJT)
    if firstPass:
        firstPass=False
        pAvgJT=avgJT
        continue;
    if abs(pAvgJT-avgJT)< epsilon:
        converged=True
    pAvgJT=avgJT
print(theta)
print("epochs ",epochs)

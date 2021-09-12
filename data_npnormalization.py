#to perform normalization of data to 0 mean and 1 std deviation
#to use: give the input file that needs to be normalized as terminal argumments
#NOTE: Output is written to stdout, so to save in a file, give apropriate filename to command via '>' to redirect stdout to that.
import sys
import pandas as pd
import numpy as np
file=open(sys.argv[1])
#read the lines from the input file
y=file.readlines()
val=[]
for x in y:
    x=x.strip('\n')
    val.append(float(x));
mean=sum(val)/len(val)
# print(mean)
stdD=0
#now calculate the mean.
for i in range(0,len(val)):
    stdD+=(val[i]-mean)**2
stdD/=len(val)
stdD=stdD**(0.5)
# print(stdD)
for i in range(0,len(val)):
    val[i]=(val[i]-mean)/stdD
for v in val:
    print(v)

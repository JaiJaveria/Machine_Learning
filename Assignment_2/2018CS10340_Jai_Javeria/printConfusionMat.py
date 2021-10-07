import numpy as np
arr=[]
try:
    f=open('confusion_matrix','rb')
except :
    print("oops! The confusion_matrix matrix doesnt seem to be present in the directory. Please run q1a and then check again.")
    exit()
arr=np.load(f)
print("The confusion matrix for part a. The rows represent actual classes and columns the predicted one")
print(arr)

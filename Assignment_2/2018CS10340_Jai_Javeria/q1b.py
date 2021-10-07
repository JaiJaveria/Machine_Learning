#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd


# In[14]:

import sys
train_file=sys.argv[1]
test_file=sys.argv[2]

data=pd.read_json(train_file, lines=True)
data=data[['overall']]

num_examples=data.shape[0]
df_overall_size=data.groupby('overall')
df_overall_size=df_overall_size['overall'].agg(['size'])
#contains the class data


# In[15]:


maj_class=df_overall_size['size'].idxmax()


# In[16]:


#training done. testing
test=pd.read_json(test_file, lines=True)
test=test[['overall']]
test_size=test.shape[0]

# In[17]:


a=0
for i in range(test_size):
    if(test.iloc[i]['overall']==maj_class):
        a+=1
print("Majiority Prediction Accuracy in percentage:", (a/test_size)*100)


# In[23]:


listClasses=(list(df_overall_size.index))
l=len(listClasses)


# In[29]:


import random
a=0
for i in range(test_size):
    k=random.randint(0,l)
    if(test.iloc[i]['overall']==k):
        a+=1
print("Random Prediction Theoretical Accuracy in percentage:", 100/l)
print("Random Prediction Actual Accuracy (would vary) in percentage:", (a/test_size)*100)


# In[ ]:





# In[ ]:

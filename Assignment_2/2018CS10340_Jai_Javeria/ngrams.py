#!/usr/bin/env python
# coding: utf-8

# In[42]:


import re
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
porter = PorterStemmer()
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
t = RegexpTokenizer(r'[a-z0-9]+')

# In[2]:


def sanitizeBasic(sent):
    #remove all punctuations, numbers and stem as well
    sent=" ".join(t.tokenize(sent))
    return sent


# In[3]:


def sanitizeStemStopWords(sent):

    sent=t.tokenize(sent)
    sent=" ".join([w for w in sent if w not in stop_words])
    sent= porter.stem(sent)
    return sent


# In[4]:


#args train_file testfile ngrams sanitize
import sys
train_file=sys.argv[1]
test_file=sys.argv[2]
#how many words to consider together
ngrams=int(sys.argv[3])
sanitize=sanitizeBasic if (int(sys.argv[4])==0)  else sanitizeStemStopWords


# In[5]:


import pandas as pd


# In[6]:


data=pd.read_json(train_file, lines=True)
data=data[['reviewText','overall']]
train_size=data.shape[0]
print("Train dataset uploaded of size: ", train_size)
#contains the class data


# In[7]:


num_classes=5


# In[8]:


#contains the class data
df_overall=data[['overall']].astype(int).copy()


# In[9]:


#find the frequencies of words in a sentence and returns a dictionary
def findFreq(sent):
    val=sent.split(" ")
    l=len(val)
    freq={}
    for i in range(l-ngrams+1):
        s=" ".join(val[i:i+ngrams])
        if s not in freq:
            freq[s]=1
        else:
            freq[s]+=1
    return freq


# In[10]:


data['reviewText']=data['reviewText'].str.lower()
# print(dat)


# In[11]:


data['reviewText']=data['reviewText'].apply(sanitize)
# print(dat)


# In[12]:


#contains the text data
dat=data[['reviewText']].copy()
# print(dat)


# In[13]:


#find the length of each sentence in data
example_length=pd.DataFrame()
example_length['length']=dat['reviewText'].apply(len)
# print(example_length)


# In[14]:


freq_df=pd.DataFrame()
freq_df['reviewText']=dat['reviewText'].apply(findFreq)
# print(freq_df)


# In[15]:


vocab=set()
num_examples=df_overall.shape[0]
for d in range(num_examples):
    for k in freq_df.iloc[d]['reviewText'].keys():
        if k not in vocab:
            vocab.add(k)
# print(len(vocab))


# In[16]:


vocab_size=len(vocab)


# In[17]:


sum_denom={} #the summation of length of reviews depending upon their class

for i in range(1,num_classes+1):
    sum_denom[i]=0
for i in range(num_examples):
    k=df_overall.iloc[i]['overall']
    sum_denom[k]+=example_length.iloc[i]['length']
# print(sum_denom)


# In[18]:


sum_numer={} #the sum of the numerator for each theta
for word in vocab:
    sum_numer[word]={}
    for i in range(1,num_classes+1):
        sum_numer[word][i]=0
for i in range(num_examples):
    k=df_overall.iloc[i]['overall']
    d=freq_df.iloc[i]['reviewText']
    for j in d:
        sum_numer[j][k]+=d[j]


# In[19]:


import math
thetas={}
for word in vocab:
    thetas[word]={}
    for i in range(1,num_classes+1):
        thetas[word][i]=math.log((sum_numer[word][i]+1)/(sum_denom[i]+vocab_size+1)) # +1 in denominator for unk token(words not in vobabulary)


# In[20]:


#calculate p(y=k)
df_overall_size=df_overall.groupby('overall')
df_overall_size=df_overall_size['overall'].agg(['size'])
df_overall_size['size']=df_overall_size['size']/num_examples
df_overall_size['size']=df_overall_size['size'].apply(math.log)


# In[21]:


#unk tokens
unk_p={i:math.log(1/(sum_denom[i]+vocab_size+1)) for i in range(1,num_classes+1)}


# In[22]:


#training done. testing
test=pd.read_json(test_file, lines=True)
test_size=test.shape[0]
print("Test dataset uploaded of size: ", test_size)

test['reviewText']=(test['reviewText'].str.lower()).apply(sanitize)


# In[23]:


def findNGrams(sent):
    val=sent.split(" ")
    l=len(val)
    ans=[]
    for i in range(l-ngrams+1):
        s=" ".join(val[i:i+ngrams])
        ans.append(s)
    return ans


# In[24]:


testR=pd.DataFrame()
testR['reviewText']=test['reviewText']
testO=pd.DataFrame()
testO['overall']=test['overall'].astype(int).copy()


# In[25]:


def predictV(setS):
    d={i:df_overall_size['size'][i] for i in range(1, num_classes+1)}
    for s in setS:
        if s not in thetas:
            for i in range(1, num_classes+1):
                d[i]+=unk_p[i]
        else:
            for i in range(1, num_classes+1):
                d[i]+=thetas[s][i]
    return max(d,key=d.get)


# In[26]:


predict=pd.DataFrame()
predict['val']=testR['reviewText'].apply(findNGrams).apply(predictV)


# In[27]:


# print(dat)
predictTrain=pd.DataFrame()
predictTrain['val']=(dat['reviewText'].apply(findNGrams)).apply(predictV)
# predictTrain['val']=dat['reviewText'].apply(uniqueWords).apply(predictV)


# In[28]:


a=0
for i in range(train_size):
    if predictTrain.iloc[i]['val']==df_overall.iloc[i]['overall']:
        a+=1
print("Training Accuracy in percentage:", (a/train_size)*100)


# In[29]:


a=0
for i in range(test_size):
    if predict.iloc[i]['val']==testO.iloc[i]['overall']:
        a+=1
print("Testing Accuracy in percentage",(a/test_size)*100)


# In[31]:


import numpy as np
def confusion_matrix(actual, predict):
    numclasses=5
    ans=np.zeros((5,5), dtype=int)
    l=len(predict)
    for i in range(l):
        ans[actual.iloc[i]['overall']-1][predict.iloc[i]['val']-1]+=1;
    return ans
if ngrams==1 and int(sys.argv[4])==0:
    c=confusion_matrix(testO, predict)
    f=open('confusion_matrix','wb')
    np.save(f, c)
    f.close()


# In[ ]:

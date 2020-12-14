#!/usr/bin/env python
# coding: utf-8

# In[130]:


# Importing the libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import string
import nltk 
import xgboost as xgb
from sklearn import metrics, model_selection, naive_bayes
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import re
from sklearn.naive_bayes import GaussianNB


warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')


# In[131]:


nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('words')


# # Data importing

# In[132]:


data = pd.read_csv("C:/Users/study gil/Documents/Reviews.csv")
print("Size of the data : ", data.shape)
# there are 568454 reviews


# In[133]:


# Use 50000 recent reviews
data = data.iloc[-50000:,]
data.index = np.arange(50000)


# In[134]:


print("Size of the data : ", data.shape)


# In[135]:


# Using Only Text
data=data[['Text','Score']] 
data['review']=data['Text']
data['rating']=data['Score']
data.drop(['Text','Score'],axis=1,inplace=True)


# In[136]:


data.head()
# distictiion between words is by 'spacing'


# In[137]:


# checking for the null values
print(data['review'].isnull().sum())
print(data['rating'].isnull().sum())  
# there is no any null value for review and rating.


# In[138]:


data.groupby('rating').count()
# counts for each level of ratings 
# we can see that ratings are skewed to the high score (positively skewd)


# In[139]:


# For binary scaling gradings with <=3 , 0 
# gradings with >3 ,1
data['rating'] = np.where(data['rating'] > 3, 1, 0)


# In[140]:


data.groupby('rating').count()
# there are 39573 postive reviews
# there are 10427 negative reviews


# # Data Preprocessing and Tokenizer

# In[141]:


# There may be some Differences in Length of Text and their Sentiments
# So Adding Variables

def get_sentiment(text):
    value = SentimentIntensityAnalyzer().polarity_scores(text)
    return value['compound']

def get_words(text):
    words = nltk.tokenize.word_tokenize(text)
    return [word for word in words if not word in string.punctuation]

data['num_words']=data['review'].apply(lambda x:len(get_words(x)))
data["sentiment"]=data["review"].apply(get_sentiment)


# In[142]:


# Split in Train set and Test set
train = data.iloc[:30000,:]
test = data.iloc[30000:,:]


# In[224]:


train.groupby('rating')['num_words','sentiment'].mean()


# In[227]:


print(set(stopwords.words('english')))


# # Logistic Regression 

# In[144]:


tfidf_vec = TfidfVectorizer(tokenizer=word_tokenize, 
                            stop_words=stopwords.words('english'), 
                            ngram_range=(1, 3), min_df=50) 
# 최소 단어 : 50
train_tfidf = tfidf_vec.fit_transform(train['review'].values.tolist())
test_tfidf = tfidf_vec.transform(test['review'].values.tolist())
train_y = train['rating']

def runLR(train_X,train_y,test_X,test_y,test_X2):
    model=LogisticRegression()
    model.fit(train_X,train_y)
    pred_test_y=model.predict_proba(test_X)
    pred_test_y2=model.predict_proba(test_X2)
    return pred_test_y, pred_test_y2, model


# In[145]:


cv_scores=[]
cols_to_drop=['review','rating']
train_X = train.drop(cols_to_drop, axis=1)
train_y=train['rating']
test_X = test.drop(cols_to_drop, axis=1)
pred_train=np.zeros([train.shape[0],2])
pred_full_test = 0

cv = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

for dev_index, val_index in cv.split(train_X,train_y):
    dev_X, val_X = train_tfidf[dev_index], train_tfidf[val_index]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    pred_val_y, pred_test_y, model = runLR(dev_X, dev_y, val_X, val_y,test_tfidf)
    pred_full_test = pred_full_test + pred_test_y
    pred_train[val_index,:] = pred_val_y
    
cv_scores.append(metrics.log_loss(val_y, pred_val_y))
print("Mean cv score For Logistic Regression with Tdif Vect: ", np.mean(cv_scores))
pred_full_test = pred_full_test / 5

train["tfidf_LR_0"] = pred_train[:,0]
train["tfidf_LR_1"] = pred_train[:,1]

test["tfidf_LR_0"] = pred_full_test[:,0]
test["tfidf_LR_1"] = pred_full_test[:,1]


# In[146]:


cvec_vec=CountVectorizer(tokenizer=word_tokenize, 
                         stop_words=stopwords.words('english'), 
                         ngram_range=(1, 3), min_df=50)
cvec_vec.fit(train['review'].values.tolist())
train_cvec = cvec_vec.transform(train['review'].values.tolist())
test_cvec = cvec_vec.transform(test['review'].values.tolist())


# In[147]:


cv_scores=[]
cols_to_drop=['review','rating']
train_X = train.drop(cols_to_drop, axis=1)
train_y=train['rating']
test_X = test.drop(cols_to_drop, axis=1)
pred_train=np.zeros([train.shape[0],2])
pred_full_test = 0

cv = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

for dev_index, val_index in cv.split(train_X,train_y):
    dev_X, val_X = train_cvec[dev_index], train_cvec[val_index]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    pred_val_y, pred_test_y, model = runLR(dev_X, dev_y, val_X, val_y,test_cvec)
    pred_full_test = pred_full_test + pred_test_y
    pred_train[val_index,:] = pred_val_y
    cv_scores.append(metrics.log_loss(val_y, pred_val_y))
print("Mean cv score For Logistic Regression with Count Vect: ", np.mean(cv_scores))
pred_full_test = pred_full_test / 5.

train["cvec_LR_0"] = pred_train[:,0]
train["cvec_LR_1"] = pred_train[:,1]

test["cvec_LR_0"] = pred_full_test[:,0]
test["cvec_LR_1"] = pred_full_test[:,1]


# # Random Forest

# In[148]:


tfidf_vec = TfidfVectorizer(tokenizer=word_tokenize, stop_words=stopwords.words('english'), ngram_range=(1, 3), min_df=50)

train_tfidf = tfidf_vec.fit_transform(train['review'].values.tolist())
test_tfidf = tfidf_vec.transform(test['review'].values.tolist())
train_y = train['rating']

def runRF(train_X,train_y,test_X,test_y,test_X2):
    model=RandomForestClassifier()
    model.fit(train_X,train_y)
    pred_test_y=model.predict_proba(test_X)
    pred_test_y2=model.predict_proba(test_X2)
    return pred_test_y, pred_test_y2, model


# In[149]:


def runRF(train_X,train_y,test_X,test_y,test_X2):
    model=RandomForestClassifier()
    model.fit(train_X,train_y)
    pred_test_y=model.predict_proba(test_X)
    pred_test_y2=model.predict_proba(test_X2)
    return pred_test_y, pred_test_y2, model


cv_scores=[]
cols_to_drop=['review','rating']
train_X = train.drop(cols_to_drop, axis=1)
train_y=train['rating']
test_X = test.drop(cols_to_drop, axis=1)
pred_train=np.zeros([train.shape[0],2])
pred_full_test = 0

cv = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

for dev_index, val_index in cv.split(train_X,train_y):
    dev_X, val_X = train_tfidf[dev_index], train_tfidf[val_index]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    pred_val_y, pred_test_y, model = runRF(dev_X, dev_y, val_X, val_y,test_tfidf)
    pred_full_test = pred_full_test + pred_test_y
    pred_train[val_index,:] = pred_val_y
    cv_scores.append(metrics.log_loss(val_y, pred_val_y))
print("Mean cv score For Random Forest with Tdif vect(char): : ", np.mean(cv_scores))
pred_full_test = pred_full_test / 5.

train["tfidf_RF_0"] = pred_train[:,0]
train["tfidf_RF_1"] = pred_train[:,1]

test["tfidf_RF_0"] = pred_full_test[:,0]
test["tfidf_RF_1"] = pred_full_test[:,1]


# In[150]:


cvec_vec=CountVectorizer(tokenizer=word_tokenize, 
                         stop_words=stopwords.words('english'),
                         ngram_range=(1, 3), min_df=50)
cvec_vec.fit(train['review'].values.tolist())
train_cvec = cvec_vec.transform(train['review'].values.tolist())
test_cvec = cvec_vec.transform(test['review'].values.tolist())


# In[151]:


cv_scores=[]
cols_to_drop=['review','rating']
train_X = train.drop(cols_to_drop, axis=1)
train_y=train['rating']
test_X = test.drop(cols_to_drop, axis=1)
pred_train=np.zeros([train.shape[0],2])
pred_full_test = 0

cv = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

for dev_index, val_index in cv.split(train_X,train_y):
    dev_X, val_X = train_cvec[dev_index], train_cvec[val_index]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    pred_val_y, pred_test_y, model = runRF(dev_X, dev_y, val_X, val_y,test_cvec)
    pred_full_test = pred_full_test + pred_test_y
    pred_train[val_index,:] = pred_val_y
    cv_scores.append(metrics.log_loss(val_y, pred_val_y))
print("Mean cv score For Random Forest with Count vect : ", np.mean(cv_scores))
pred_full_test = pred_full_test / 5.

train["cvec_RF_0"] = pred_train[:,0]
train["cvec_RF_1"] = pred_train[:,1]

test["cvec_RF_0"] = pred_full_test[:,0]
test["cvec_RF_1"] = pred_full_test[:,1]


# # Naive Bayes 

# In[153]:


tfidf_vec = TfidfVectorizer(tokenizer=word_tokenize, stop_words=stopwords.words('english'), ngram_range=(1, 3), min_df=50)

train_tfidf = tfidf_vec.fit_transform(train['review'].values.tolist())
test_tfidf = tfidf_vec.transform(test['review'].values.tolist())
train_y = train['rating']

def runNB(train_X,train_y,test_X,test_y,test_X2):
    model=GaussianNB()
    model.fit(train_X,train_y)
    pred_test_y=model.predict_proba(test_X)
    pred_test_y2=model.predict_proba(test_X2)
    return pred_test_y, pred_test_y2, model


# In[154]:


cv_scores=[]
cols_to_drop=['review','rating']
train_X = train.drop(cols_to_drop, axis=1)
train_y=train['rating']
test_X = test.drop(cols_to_drop, axis=1)
pred_train=np.zeros([train.shape[0],2])
pred_full_test = 0

cv = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=0)


for dev_index, val_index in cv.split(train_X,train_y):
    dev_X, val_X = train_tfidf[dev_index].toarray(), train_tfidf[val_index].toarray()
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    pred_val_y, pred_test_y, model = runNB(dev_X, dev_y, val_X, val_y, test_tfidf.toarray())
    pred_full_test = pred_full_test + pred_test_y
    pred_train[val_index,:] = pred_val_y
    cv_scores.append(metrics.log_loss(val_y, pred_val_y))
print("Mean cv score For Naive Bayes with Tdif Vect: ", np.mean(cv_scores))
pred_full_test = pred_full_test / 5.


train["tfidf_nb_0"] = pred_train[:,0]
train["tfidf_nb_1"] = pred_train[:,1]

test["tfidf_nb_0"] = pred_full_test[:,0]
test["tfidf_nb_1"] = pred_full_test[:,1]


# In[155]:


cvec_vec=CountVectorizer(tokenizer=word_tokenize, stop_words=stopwords.words('english'), ngram_range=(1, 3), min_df=50)
cvec_vec.fit(train['review'].values.tolist())
train_cvec = cvec_vec.transform(train['review'].values.tolist())
test_cvec = cvec_vec.transform(test['review'].values.tolist())


# In[156]:


cv_scores=[]
cols_to_drop=['review','rating']
train_X = train.drop(cols_to_drop, axis=1)
train_y=train['rating']
test_X = test.drop(cols_to_drop, axis=1)
pred_train=np.zeros([train.shape[0],2])
pred_full_test = 0

cv = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=0)


for dev_index, val_index in cv.split(train_X,train_y):
    dev_X, val_X = train_cvec[dev_index].toarray(), train_cvec[val_index].toarray()
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    pred_val_y, pred_test_y, model = runNB(dev_X, dev_y, val_X, val_y, test_cvec.toarray())
    pred_full_test = pred_full_test + pred_test_y
    pred_train[val_index,:] = pred_val_y
    cv_scores.append(metrics.log_loss(val_y, pred_val_y))
print("Mean cv score For Naive Bayes with Count Vect: ", np.mean(cv_scores))
pred_full_test = pred_full_test / 5.


train["cvec_nb_0"] = pred_train[:,0]
train["cvec_nb_1"] = pred_train[:,1]

test["cvec_nb_0"] = pred_full_test[:,0]
test["nb_cvec_char_1"] = pred_full_test[:,1]


# In[169]:


train_y


# In[222]:


cols_to_drop=['review','rating']
train_X = train.drop(cols_to_drop, axis=1)
train_y=train['rating']
test_index = test['rating'].values
test_X = test.drop(cols_to_drop, axis=1)

params = {}
params['objective'] = 'binary:logistic'
params['max_depth'] = 4
params['min_child_weight'] = 1
params['eta'] = 0.1
params['silent'] = 1
params['eval_metric'] = "mlogloss"
params['subsample'] = 0.8
params['colsample_bytree'] = 0.3
params['seed'] = 0

model = xgb.XGBClassifier(param = params)
model.fit(train_X,train_y)
xgb_pred = list(model.predict(test_X))


# In[223]:


metrics.accuracy_score(test_index, xgb_pred)


# In[229]:


fig, ax = plt.subplots(figsize=(12,12))
xgb.plot_importance(model, max_num_features=20, height=0.8, ax=ax)
plt.show()


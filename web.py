#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pickle as pkl
from sklearn.linear_model import LogisticRegression
from pickle import load
from pickle import dump
import streamlit as st
import pandas as pd


# In[8]:


st.title("Prediction of Survival")
st.sidebar.header("User Input Parameters")

def user_input_features():
    Pclass=st.sidebar.selectbox('PassengerClass',('1','2','3'))
    Age=st.sidebar.number_input("Age")
    SibSp=st.sidebar.selectbox("SibSp",('0','1','2','3','4','5','8'))
    Parch=st.sidebar.selectbox("Parch",('0','1','2','3','4','5','6'))
    Fare=st.sidebar.number_input("Enter the Fare:")
    Sex=st.sidebar.selectbox("Gender(0-Female,1-Male)",('0','1'))
    Embarked=st.sidebar.selectbox("Embarked(0-C,1-Q,2-S)",("0","1","2"))
    data={'Pclass':Pclass,
          "Sex":Sex,
          "Age":Age,
          "SibSp":SibSp,
          "Parch":Parch,
          "Fare":Fare,
          "Embarked":Embarked}
    features=pd.DataFrame(data,index=[0])
    return features

df=user_input_features()

st.subheader("User Input Parameters")
st.write(df)

# load the model from disk
# predict
loaded_model=load(open("deploy.pkl",'rb'))


prediction=loaded_model.predict(df)
prediction_proba=loaded_model.predict_proba(df)

st.subheader('Predicted Result')
st.write("You Will Survive.." if prediction[0]==1 else "There is no Chances to survive!.")


st.subheader('Prediction Probability')
st.write(prediction_proba)


# In[ ]:





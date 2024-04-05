# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 18:52:36 2024

@author: DELL
"""

import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
bankruptcy=pd.read_csv('C:\Pandas//bankruptcy_prevention.csv',sep=';', header = 0)
#bankruptcy.drop([' class'],inplace=True,axis=1)
bankruptcy=bankruptcy.dropna()
#inpute variable
x=bankruptcy.iloc[:,:6]
x.dropna()
#target variable
y=bankruptcy[' class']
clf = LogisticRegression()
clf.fit(x,y)
missing_values = x.isnull().sum()

st.title('Model Deployment: Logistic Regression')
st.sidebar.header('User Input Parameters')
# User input for features
def get_input_with_selection(label):
    selection_type = st.sidebar.radio(f"Select {label}:", ("Input", "Selection"))
    if selection_type == "Input":
        value = st.sidebar.number_input(f"{label} (Numeric Value)", min_value=0.0, max_value=1.0, step=0.1)
    else:
        options = [0.0, 0.5, 1.0]
        value = st.sidebar.selectbox(f"{label} (Selection)", options)
    return value


industrial_risk = get_input_with_selection("Industrial Risk")
management_risk = get_input_with_selection("Management Risk")
financial_flexibility = get_input_with_selection("Financial Flexibility")
credibility = get_input_with_selection("Credibility")
competitiveness = get_input_with_selection("Competitiveness")
operating_risk = get_input_with_selection("Operating Risk")
# Function to make predictions
def predict_bankruptcy(industrial_risk, management_risk, financial_flexibility, credibility, competitiveness, operating_risk):
    
    industrial_risk = float(industrial_risk)
    management_risk = float(management_risk)
    financial_flexibility = float(financial_flexibility)
    credibility = float(credibility)
    competitiveness = float(competitiveness)
    operating_risk = float(operating_risk)
    
    features = [[industrial_risk, management_risk, financial_flexibility, credibility, competitiveness, operating_risk]]
    prediction = clf.predict(features)
    return prediction

# Predict bankruptcy risk
if st.button("Predict"):
    prediction = predict_bankruptcy(industrial_risk, management_risk, financial_flexibility, credibility, competitiveness, operating_risk)
    st.write("Prediction:", prediction)
import pandas as pd
import numpy as np
import pickle
import streamlit as st

df = pd.read_csv("Dataset.csv")
df = df.drop(columns = ["BodyFat"])

st.sidebar.header("Variables' Description")
btn_1 = st.sidebar.button("Description")

st.title("Body Fat Percentage")

st.header("Dataset")
st.write(df.head())

st.sidebar.header("Inputs Giving")

if btn_1 :
    st.header("Variables' Descriptions")
    st.markdown("* **Density determined from underwater weighing**")
    st.markdown("* **Percent body fat from Siri's (1956) equation**")
    st.markdown("* **Age (years)**")
    st.markdown("* **Weight (lbs)**")
    st.markdown("* **Height (inches)**")
    st.markdown("* **Neck circumference (cm)**")
    st.markdown("* **Chest circumference (cm)**")
    st.markdown("* **Abdomen 2 circumference (cm)**")
    st.markdown("* **Hip circumference (cm)**")
    st.markdown("* **Thigh circumference (cm)**")
    st.markdown("* **Knee circumference (cm)**")
    st.markdown("* **Ankle circumference (cm)**")
    st.markdown("* **Biceps (extended) circumference (cm)**")
    st.markdown("* **Forearm circumference (cm)**")
    st.markdown("* **Wrist circumference (cm)**")
    
    
Density = st.sidebar.slider("Please select input for the Density vairable" , min_value = 0.995 , max_value = 1.1089 , step = 0.001)
Age = st.sidebar.slider("Please select input for the Age vairable" , min_value = 22 , max_value = 81 , step = 1)
Weight = st.sidebar.slider("Please select input for the Weight vairable" , min_value = 118.5 , max_value = 363.15 , step = 0.1)
Height = st.sidebar.slider("Please select input for the Height vairable" , min_value = 29.5 , max_value = 77.75 , step = 0.25)
Neck = st.sidebar.slider("Please select input for the Neck vairable" , min_value = 31.1 , max_value = 51.2 , step = 0.1)
Chest = st.sidebar.slider("Please select input for the Chest vairable" , min_value = 79.3 , max_value = 136.2 , step = 0.1)
Abdomen = st.sidebar.slider("Please select input for the Abdomen vairable" , min_value = 69.4 , max_value = 148.1 , step = 0.1)
Hip = st.sidebar.slider("Please select input for the Hip vairable" , min_value = 85.0 , max_value = 147.7 , step = 0.1)
Thigh = st.sidebar.slider("Please select input for the Thigh vairable" , min_value = 47.2 , max_value = 87.3 , step = 0.1)
Knee = st.sidebar.slider("Please select input for the Knee vairable" , min_value = 33.0 , max_value = 49.1 , step = 0.1)
Ankle = st.sidebar.slider("Please select input for the Ankle vairable" , min_value = 19.1 , max_value = 33.9 , step = 0.1)
Biceps = st.sidebar.slider("Please select input for the Biceps vairable" , min_value = 24.8 , max_value = 45.0 , step = 0.1)
Forearm = st.sidebar.slider("Please select input for the Forearm vairable" , min_value = 21.0 , max_value = 34.9 , step = 0.1)
Wrist = st.sidebar.slider("Please select input for the Wrist vairable" , min_value = 15.8 , max_value = 21.4 , step = 0.1)

btn_2 = st.sidebar.button("Predict")

if btn_2 : 
    st.header("You have entered these inputs")
    data_input = pd.DataFrame(data = {"Density" : [Density], "Age" : [Age],
                                  "Weight" : [Weight] , "Height" : [Height],
                                  "Neck" : [Neck] , "Chest" : [Chest],
                                  "Abdomen" : [Abdomen] , "Hip" : [Hip],
                                  "Thigh" : [Thigh] , "Knee" : [Knee],
                                  "Ankle" : [Ankle] , "Biceps" : [Biceps],
                                  "Forearm" : [Forearm] , "Wrist" : [Wrist]})
    st.write(data_input)
    
    
    model = pickle.load(open("Model" , "rb"))
    
    y_pred = model.predict(data_input)
    
    st.header("Prediction")
    st.markdown(f"The percentage of the Body Fat is **{int(y_pred)}%**")

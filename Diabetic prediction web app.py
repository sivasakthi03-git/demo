# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 20:30:46 2025

@author: murug
"""

import numpy as np
import pickle
import streamlit as st

loaded_model = pickle.load(open("C:/Users/murug/OneDrive/Desktop/deployment/streamlit_diabetes/trained_model.sav", 'rb'))

# creating function for prediction

def diabetes_prediction(input_data):

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
      return 'The person is not diabetic'
    else:
      return 'The person is diabetic'
  
def main():
    
    # giving title 
    st.title('Diabetic Predictive Web App')
    # input from user 
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose     = st.text_input('glucose level')
    BloodPressure = st.text_input('blood pressure value')
    SkinThickness = st.text_input('skin thickness value')
    Insulin  = st.text_input('insulin level')
    BMI   = st.text_input('BMI level')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    Age = st.text_input('age of the person')
    # code for prediction
    diagnosis = ''
    # create button for prediction
    if st.button('diabetics test result'):
        diagnosis = diabetes_prediction([Pregnancies,Glucose
                                         ,BloodPressure,
                                         SkinThickness,
                                         Insulin,
                                         BMI,
                                         DiabetesPedigreeFunction,
                                         Age])
    st.success(diagnosis)
    
  
    
if __name__ == '__main__':
    main()
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
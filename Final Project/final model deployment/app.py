# Importing all necessary libraries
import streamlit as st
import pickle
import numpy as np

# Loading the saved Model
model = pickle.load(open("final_model.pkl", "rb"))



def predict_default(features):

    features = np.array(features).astype(np.float64).reshape(1,-1)
    
    prediction = model.predict(features)
    probability = model.predict_proba(features)

    return prediction, probability

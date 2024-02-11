# Run it with 'streamlit run main.py
# Import libraries that you need
import streamlit as st
import pandas as pd
import pickle
from math import ceil

# Load the model
model = pickle.load(open(r'models\\model_1A.pkl', 'rb'))

# Add your title and description
st.title("Tjoean's Sales Prediction")
st.markdown("Here we are using month as the input to predict the sales on that month")

# Get the input
st.subheader("Enter the month")
month = st.text_input('', '1')  # Default value as a string '1'

# Prediction
st.subheader(f"Predicted Sales on month-{month}")
prediction = model.predict([[int(month)]])  # Make sure to convert the input to int
# Apply ceil to each element in the prediction array and join them into a string
ceil_prediction = ' '.join([str(ceil(value)) for value in prediction[0]])
st.code(ceil_prediction)
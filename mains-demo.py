import streamlit as st
import pickle
import pandas as pd
import plotly.express as px
from math import ceil

# Define a function to perform prediction
def perform_prediction(month):
    prediction = model.predict([[month]])
    predicted_values = [ceil(value) for value in prediction[0]]
    return predicted_values

# Define a function to add the predicted sales data as points and lines
def add_prediction_lines(fig, df, month, predicted_values):
    for i, col in enumerate(['Shumai 10 Pcs', 'Shumai 20 Pcs', 'Shumai 30 Pcs', 'Chicken Lumpia 10 Pcs']):
        fig.add_scatter(x=[df['Month'].iloc[-1], month], y=[df[col].iloc[-1], predicted_values[i]],
                        mode='lines+markers+text', text=[None, predicted_values[i]],
                        textposition='top center', name=f'Predicted {col}')

# Define a function to add only the predicted points
def add_prediction_points(fig, month, predicted_values):
    for i, col in enumerate(['Shumai 10 Pcs', 'Shumai 20 Pcs', 'Shumai 30 Pcs', 'Chicken Lumpia 10 Pcs']):
        fig.add_scatter(x=[month], y=[predicted_values[i]], mode='markers+text',
                        text=[predicted_values[i]], textposition='top center', name=f'Predicted {col}')

# Load the model
model = pickle.load(open('models/model_3B.pkl', 'rb'))

# Load your dataset as a pandas DataFrame
df = pd.read_excel('data/dataModeling/dataModeling_DimsumTJOEAN.xlsx')

# Initialize the Streamlit app
st.title("Tjoean's Sales Prediction")
st.markdown("This website have a purpose for predicting the sales figures of products sold by Tjoean.id")
st.markdown("Disclaimer: The machine cannot guarantee that it can predict well, because it is a simulation.")
# Display the DataFrame in an expandable section
with st.expander("View Sales Data"):
    st.dataframe(df)

# Plot the line chart for the existing sales data using plotly.express
fig = px.line(
    df,
    x='Month',
    y=['Shumai 10 Pcs', 'Shumai 20 Pcs', 'Shumai 30 Pcs', 'Chicken Lumpia 10 Pcs'],
    labels={'value': 'Sales', 'variable': 'Products'},
    title='Monthly Sales Data'
)
st.plotly_chart(fig, use_container_width=True)

# Input for prediction
st.markdown("## Enter the month to predict the sales")
st.write('Please input month sales number from range 1-40')
month = st.number_input('Month Number', min_value=1, value=1, step=1)

# Confirmation button
if st.button('Predict Sales') or 'predict_button' in st.session_state:
    predicted_values = perform_prediction(month)

    # Update figure with predictions
    if month > df['Month'].max():
        # Add lines for predictions if months above predicted_values
        add_prediction_lines(fig, df, month, predicted_values)
    else:
        # Add only points for predictions if months below or same on predicted_values
        add_prediction_points(fig, month, predicted_values)

    # Plot the updated figure
    st.plotly_chart(fig, use_container_width=True)

    # Display predicted sales values
    cell_col, disp_col = st.columns([0.6, 0.4])
    cell_col.subheader(f"Predicted Sales for Month {month}:")
    for i, col in enumerate(['Shumai 10 Pcs', 'Shumai 20 Pcs', 'Shumai 30 Pcs', 'Chicken Lumpia 10 Pcs']):
        disp_col.metric(label=col, value=predicted_values[i])
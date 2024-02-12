import streamlit as st
import pickle
import pandas as pd
import plotly.express as px
from math import ceil

# Define a function to draw the line chart
def plot_line_chart(data, title='Monthly Sales Data'):
    fig = px.line(
        data,
        x='Month',
        y=['Shumai 10 Pcs', 'Shumai 20 Pcs', 'Shumai 30 Pcs', 'Chicken Lumpia 10 Pcs'],
        labels={'value': 'Sales', 'variable': 'Products'},
        title=title
    )
    st.plotly_chart(fig, use_container_width=True)
    return fig

# Define a function to perform prediction
def perform_prediction(month):
    prediction = model.predict([[month]])
    predicted_values = [ceil(value) for value in prediction[0]]
    return predicted_values

# Define a function to add the predicted sales data as points and lines
def add_prediction_lines(fig, df, month, predicted_values):
    if month > df['Month'].max():
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

# Page Configuration 
# To define settings for the app by giving it a page title and icon that are displayed on the browser
st.set_page_config(
    page_title="Tjoean's Sales Prediction (v2A)",
    page_icon="🍴",
    layout="wide",
    initial_sidebar_state="expanded")

# Initialize the Streamlit app
st.title("Tjoean's Sales Prediction")
st.markdown("""
This website is designed to predict the sales figures of products sold by Tjoean.id.
""")
st.markdown("""
**Disclaimer**: The predictions are simulations and cannot guarantee accuracy.
""")

# Data Source Selection
st.markdown("## Data Source Selection")

# Allow the user to choose between using default data or uploading their own
data_source = st.radio("Choose the data source for prediction:",
                       ('Use default data', 'Upload my own data'))

if data_source == 'Use default data':
    # Load your dataset as a pandas DataFrame
    df = pd.read_excel('data/dataModeling/dataModeling_DimsumTJOEAN.xlsx')
    # Display the DataFrame in an expandable section
    with st.expander("View Sales Data"):
        st.dataframe(df)
    # Plot the line chart for the existing sales data using plotly.express
    fig = plot_line_chart(df)

elif data_source == 'Upload my own data':
    uploaded_file = st.file_uploader("Upload your Excel file. Please ensure it contains the required columns: DATE, Month, Shumai 10 Pcs, Shumai 20 Pcs, Shumai 30 Pcs, Chicken Lumpia 10 Pcs", type=['xlsx'])
    if uploaded_file is not None:
        try:
            df_uploaded = pd.read_excel(uploaded_file)
            # Check if the required columns are in the uploaded file
            required_columns = ['DATE', 'Month', 'Shumai 10 Pcs', 'Shumai 20 Pcs', 'Shumai 30 Pcs', 'Chicken Lumpia 10 Pcs']
            if not all(col in df_uploaded.columns for col in required_columns):
                st.error("The uploaded file does not contain the required columns.")
            else:
                df = df_uploaded  # Use the uploaded data
                st.success("File uploaded successfully!")
                # Display the DataFrame in an expandable section
                with st.expander("View Sales Data"):
                    st.dataframe(df)
                fig = plot_line_chart(df, title='Uploaded Sales Data')
        except Exception as e:
            st.error(f"An error occurred: {e}")

# Input for prediction
st.markdown("## Enter the month to predict the sales")
month = st.number_input('Month Number', min_value=1, value=1, step=1)

# Perform prediction on button press
if st.button('Predict Sales'):
    if 'df' not in locals():
        st.error("Please upload data to predict sales.")
    elif month > df['Month'].max():
        predicted_values = perform_prediction(month)
        # Update the line chart with the predicted values
        add_prediction_lines(fig, df, month, predicted_values)
        st.plotly_chart(fig, use_container_width=True)
        # Display predicted sales values
        st.subheader(f"Predicted Sales for Month {month}:")
        for i, col in enumerate(['Shumai 10 Pcs', 'Shumai 20 Pcs', 'Shumai 30 Pcs', 'Chicken Lumpia 10 Pcs']):
            st.metric(col, predicted_values[i])
    else:
        predicted_values = perform_prediction(month)
        # Update the point chart for predict if months below or same on predicted_values
        add_prediction_points(fig, month, predicted_values)
        st.plotly_chart(fig, use_container_width=True)
        # Display predicted sales values
        st.subheader(f"Predicted Sales for Month {month}:")
        for i, col in enumerate(['Shumai 10 Pcs', 'Shumai 20 Pcs', 'Shumai 30 Pcs', 'Chicken Lumpia 10 Pcs']):
            st.metric(col, predicted_values[i])
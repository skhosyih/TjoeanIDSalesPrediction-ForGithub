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
    page_title="Tjoean's Sales Prediction (v2B)",
    page_icon="🍴",
    layout="wide",
    initial_sidebar_state="expanded")

# Initialize the Streamlit app
st.title("Tjoean's Sales Prediction")
st.markdown("This website is designed to predict the sales figures of products sold by Tjoean.id")
st.markdown("**Disclaimer**: Predictions are simulations and cannot guarantee accuracy.")

# Sidebar for Data Source Selection
st.sidebar.markdown("## Data Source Selection")
data_source = st.sidebar.selectbox(
    "Choose the data source for prediction:",
    ('Use default data', 'Upload my own data'),
    key='data_source'
)

# Initialize state for file uploader
if 'upload_data' not in st.session_state:
    st.session_state['upload_data'] = False

# Initialize state for storing DataFrame
if 'df' not in st.session_state:
    st.session_state.df = pd.read_excel('data/dataModeling/dataModeling_DimsumTJOEAN.xlsx')

# Show file uploader if 'Upload my own data' is selected
if data_source == 'Upload my own data':
    st.session_state['upload_data'] = True
    uploaded_file = st.sidebar.file_uploader(
        "Upload your Excel file. Please ensure it contains the required columns: DATE, Month, Shumai 10 Pcs, Shumai 20 Pcs, Shumai 30 Pcs, Chicken Lumpia 10 Pcs",
        type=['xlsx'],
        key='file_uploader'
    )
    if uploaded_file is not None:
        try:
            df_uploaded = pd.read_excel(uploaded_file)
            required_columns = ['DATE', 'Month', 'Shumai 10 Pcs', 'Shumai 20 Pcs', 'Shumai 30 Pcs', 'Chicken Lumpia 10 Pcs']
            if not all(col in df_uploaded.columns for col in required_columns):
                st.error("The uploaded file does not contain the required columns.")
            else:
                st.session_state.df = df_uploaded  # Use the uploaded data
                st.success("File uploaded successfully!")
        except Exception as e:
            st.error(f"An error occurred: {e}")
elif data_source == 'Use default data':
    # Always load and show default data if that option is selected
    st.session_state.df = pd.read_excel('data/dataModeling/dataModeling_DimsumTJOEAN.xlsx')

# Always show data in an expander regardless of the button state
with st.expander("View Sales Data"):
    st.dataframe(st.session_state.df)

# Sidebar for Month Selection
st.sidebar.markdown("## Enter the month to predict the sales")
month = st.sidebar.number_input('Month Number', min_value=1, value=1, step=1, key='month_input')

# Predict button (always shown in sidebar)
if st.sidebar.button('Predict Sales'):
    st.session_state['predict_button'] = True

# Main page display logic based on whether predict button was clicked
if 'predict_button' in st.session_state and st.session_state['predict_button']:
    df = st.session_state.df
    predicted_values = perform_prediction(month)
    fig = plot_line_chart(df)  # Redraw chart to include the prediction
    
    if month > df['Month'].max():
        add_prediction_lines(fig, df, month, predicted_values)
    else:
        add_prediction_points(fig, month, predicted_values)
    
    st.plotly_chart(fig, use_container_width=True)
    st.subheader(f"Predicted Sales for Month {month}:")
    for i, col in enumerate(['Shumai 10 Pcs', 'Shumai 20 Pcs', 'Shumai 30 Pcs', 'Chicken Lumpia 10 Pcs']):
        st.metric(col, predicted_values[i])
    
    # Reset the predict button state after handling prediction
    st.session_state['predict_button'] = False
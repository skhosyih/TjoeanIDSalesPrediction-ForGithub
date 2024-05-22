import streamlit as st
import pickle
import pandas as pd
import plotly.express as px
from math import ceil

# Load the model
model = pickle.load(open('models/model_1A.pkl', 'rb'))

# Page Configuration 
st.set_page_config(
    page_title="Tjoean's Sales Prediction",
    page_icon="🍴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define all functions at the top
# Define a function to draw the line chart
def plot_line_chart(data, title='Monthly Sales Data'):
    fig = px.line(
        data,
        x='Month',
        y=['Shumai 10 Pcs', 'Shumai 20 Pcs', 'Shumai 30 Pcs', 'Chicken Lumpia 10 Pcs'],
        labels={'value': 'Sales', 'variable': 'Products'},
        title=title
    )
    return fig

# Define a function to perform prediction
def perform_prediction(month):
    prediction = model.predict([[month]])
    return [ceil(value) for value in prediction[0]]

# Define a function to add only the predicted points and or to add the predicted sales data as points and lines
def add_prediction_to_chart(fig, df, month, predicted_values, extend_line=False):
    products = ['Shumai 10 Pcs', 'Shumai 20 Pcs', 'Shumai 30 Pcs', 'Chicken Lumpia 10 Pcs']
    for i, col in enumerate(products):
        x_values = [month] if not extend_line else [df['Month'].iloc[-1], month]
        y_values = [predicted_values[i]] if not extend_line else [df[col].iloc[-1], predicted_values[i]]
        fig.add_scatter(
            x=x_values, y=y_values, mode='lines+markers+text',
            text=[None, predicted_values[i]] if extend_line else [predicted_values[i]],
            textposition='top center', name=f'Predicted {col}'
        )

# Main function to run the app
def main():
    st.sidebar.title("🍴 Tjoean's Sales Prediction")
    
    st.sidebar.write("#####") # Give some space
    
    st.sidebar.markdown("This website is designed to predict the sales figures of products sold by Tjoean.id")
    st.sidebar.markdown("**Disclaimer**: Predictions are simulations and cannot guarantee accuracy.")
    
    st.sidebar.write("#####") # Give some space
    
    df = pd.read_excel('data/dataModeling/dataModeling_DimsumTJOEAN.xlsx')

    with st.expander("View Sales Data"):
        st.dataframe(df)
    
    month = st.sidebar.number_input('Enter the month to predict the sales', min_value=1, value=1, step=1)
    if st.sidebar.button('Predict Sales'):
        predicted_values = perform_prediction(month)
        fig = plot_line_chart(df)
        
        # Define a variable to add the predicted sales data as points and lines
        extend_line = month > df['Month'].max()
        
        add_prediction_to_chart(fig, df, month, predicted_values, extend_line)
        st.plotly_chart(fig, use_container_width=True)
        st.subheader(f"Predicted Sales for Month {month}:")
        
        for i, col in enumerate(['Shumai 10 Pcs', 'Shumai 20 Pcs', 'Shumai 30 Pcs', 'Chicken Lumpia 10 Pcs']):
            st.metric(col, predicted_values[i])

if __name__ == "__main__":
    main()
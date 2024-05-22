import streamlit as st
import pickle
import pandas as pd
import plotly.express as px
from math import ceil
from fpdf import FPDF
import tempfile

# Load the model
model = pickle.load(open(r'models/model_1A.pkl', 'rb'))

# Page Configuration 
st.set_page_config(
    page_title="Tjoean's Sales Prediction",
    page_icon="🍴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define all functions at the top
# Define a function to draw the bar chart
def plot_bar_chart(data, title='Monthly Sales Data'):
    melted_data = data.melt(id_vars='Month', value_vars=['Shumai 10 Pcs', 'Shumai 20 Pcs', 'Shumai 30 Pcs', 'Chicken Lumpia 10 Pcs'],
                            var_name='Product', value_name='Sales')
    fig = px.bar(
        melted_data,
        x='Month',
        y='Sales',
        color='Product',
        barmode='group',
        title=title
    )
    fig.update_layout(
        xaxis=dict(tickmode='linear'),
        bargap=0.15,
        bargroupgap=0.1
    )
    fig.update_xaxes(tickangle=-45)
    fig.update_yaxes(automargin=True)
    fig.update_layout(width=1200, height=600)
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

# Define a function to generate a PDF report
def generate_pdf_report(df, predicted_values, month):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Add a title
    pdf.cell(200, 10, txt="Tjoean's Sales Prediction Report", ln=True, align='C')
    
    # Add the sales data table
    pdf.ln(10)
    pdf.cell(200, 10, txt="Sales Data:", ln=True)
    
    for i in range(len(df)):
        row = df.iloc[i]
        pdf.cell(200, 10, txt=f"Month: {row['Month']}, Shumai 10 Pcs: {row['Shumai 10 Pcs']}, Shumai 20 Pcs: {row['Shumai 20 Pcs']}, Shumai 30 Pcs: {row['Shumai 30 Pcs']}, Chicken Lumpia 10 Pcs: {row['Chicken Lumpia 10 Pcs']}", ln=True)
    
    # Add the predicted sales table
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Predicted Sales for Month {month}:", ln=True)
    products = ['Shumai 10 Pcs', 'Shumai 20 Pcs', 'Shumai 30 Pcs', 'Chicken Lumpia 10 Pcs']
    for i, col in enumerate(products):
        pdf.cell(200, 10, txt=f"{col}: {predicted_values[i]}", ln=True)
    
    # Add previous and next month predictions
    previous_month = month - 1
    next_month = month + 1
    
    if previous_month > 0:
        previous_predicted_values = perform_prediction(previous_month)
        pdf.ln(10)
        pdf.cell(200, 10, txt=f"Predicted Sales for Previous Month ({previous_month}):", ln=True)
        for i, col in enumerate(products):
            pdf.cell(200, 10, txt=f"{col}: {previous_predicted_values[i]}", ln=True)
    
    next_predicted_values = perform_prediction(next_month)
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Predicted Sales for Next Month ({next_month}):", ln=True)
    for i, col in enumerate(products):
        pdf.cell(200, 10, txt=f"{col}: {next_predicted_values[i]}", ln=True)
    
    # Save the PDF to a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    pdf.output(temp_file.name)
    
    return temp_file.name

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
        fig = plot_bar_chart(df)
        
        # Define a variable to add the predicted sales data as points and lines
        extend_line = month > df['Month'].max()
        
        add_prediction_to_chart(fig, df, month, predicted_values, extend_line)
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader(f"Predicted Sales for Month {month}:")
        prediction_df = pd.DataFrame({
            'Product': ['Shumai 10 Pcs', 'Shumai 20 Pcs', 'Shumai 30 Pcs', 'Chicken Lumpia 10 Pcs'],
            'Predicted Sales': predicted_values
        })
        st.table(prediction_df)
        
        # Generate PDF report
        if st.sidebar.button('Generate PDF Report'):
            pdf_path = generate_pdf_report(df, predicted_values, month)
            with open(pdf_path, "rb") as pdf_file:
                st.sidebar.download_button(
                    label="Download PDF Report",
                    data=pdf_file,
                    file_name="sales_prediction_report.pdf",
                    mime="application/pdf"
                )

if __name__ == "__main__":
    main()
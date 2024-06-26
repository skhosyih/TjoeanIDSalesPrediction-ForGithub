import streamlit as st
import pickle
import pandas as pd
import plotly.express as px
from math import ceil
import matplotlib.pyplot as plt
import tempfile
from fpdf import FPDF

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
# Define a function to draw the bar chart with wider bars and enable scrolling
def plot_bar_chart(data, title='Monthly Sales Data'):
    melted_data = data.melt(id_vars='Month', value_vars=['Shumai 10 Pcs', 'Shumai 20 Pcs', 'Shumai 30 Pcs', 'Chicken Lumpia 10 Pcs'],
                            var_name='Product', value_name='Sales')
    fig = px.bar(
        melted_data,
        x='Month',
        y='Sales',
        color='Product',
        barmode='group',
        title=title,
        text='Sales'
    )
    fig.update_layout(
        xaxis=dict(tickmode='linear', dtick=1),
        bargap=0.05,  # reduce gap between bars
        bargroupgap=0.05,  # reduce gap between groups
        width=1800,  # make the figure wider
        height=600
    )
    fig.update_xaxes(range=[0, 10])  # show only the first 10 months by default
    fig.update_yaxes(automargin=True)
    return fig

def save_bar_chart(data, filename, title='Chart', xaxis_title='Product', yaxis_title='Sales'):
    plt.figure(figsize=(10, 6))
    bars = plt.bar(data['Product'], data['Sales'], color=['blue', 'orange', 'green', 'red'])
    plt.xlabel(xaxis_title)
    plt.ylabel(yaxis_title)
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    
    # Adding labels on the bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, int(yval), va='bottom')  # va: vertical alignment
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Define a function to perform prediction
def perform_prediction(month):
    prediction = model.predict([[month]])
    return [ceil(value) for value in prediction[0]]

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
    
    # Table Header
    pdf.set_font('Arial', 'B', 9)
    headers = ['Month', 'Shumai 10 Pcs', 'Shumai 20 Pcs', 'Shumai 30 Pcs', 'Chicken Lumpia 10 Pcs']
    col_width = pdf.w / 5.5  # Equal width for each column
    for header in headers:
        pdf.cell(col_width, 10, header, border=1)
    pdf.ln(10)
    
    # Table contents
    pdf.set_font('Arial', '', 9)
    for i in range(len(df)):
        pdf.cell(col_width, 10, str(df['Month'].iloc[i]), border=1)
        pdf.cell(col_width, 10, str(df['Shumai 10 Pcs'].iloc[i]), border=1)
        pdf.cell(col_width, 10, str(df['Shumai 20 Pcs'].iloc[i]), border=1)
        pdf.cell(col_width, 10, str(df['Shumai 30 Pcs'].iloc[i]), border=1)
        pdf.cell(col_width, 10, str(df['Chicken Lumpia 10 Pcs'].iloc[i]), border=1)
        pdf.ln(10)
    
    # Add the predicted sales table
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Predicted Sales for Month {month}:", ln=True)
    
    products = ['Shumai 10 Pcs', 'Shumai 20 Pcs', 'Shumai 30 Pcs', 'Chicken Lumpia 10 Pcs']
    
    # Table Header for Predicted Sales
    pdf.set_font('Arial', 'B', 9)
    headers = ['Shumai 10 Pcs', 'Shumai 20 Pcs', 'Shumai 30 Pcs', 'Chicken Lumpia 10 Pcs']
    for header in headers:
        pdf.cell(col_width, 10, header, border=1)
    pdf.ln(10)
    
    # Table contents for Predicted Sales
    pdf.set_font('Arial', '', 9)
    for i, col in enumerate(products):
        pdf.cell(col_width, 10, str(predicted_values[i]), border=1)
    pdf.ln(10)

    # Add the predicted sales chart image
    predicted_chart_path = tempfile.NamedTemporaryFile(delete=False, suffix='.png').name
    
    # Save the predicted sales bar chart
    save_bar_chart(
        data=pd.DataFrame({
            'Product': ['Shumai 10 Pcs', 'Shumai 20 Pcs', 'Shumai 30 Pcs', 'Chicken Lumpia 10 Pcs'],
            'Sales': predicted_values
        }),
        filename=predicted_chart_path,
        title='Predicted Sales Data'
    )
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Predicted Sales Data Chart for Month {month}:", ln=True)
    pdf.image(predicted_chart_path, x=10, y=pdf.get_y(), w=pdf.w - 20)
    pdf.ln(85)
    
    # Add previous and next month predictions
    previous_month = month - 1
    next_month = month + 1
    
    if previous_month > 0:
        previous_predicted_values = perform_prediction(previous_month)
        pdf.ln(50)
        pdf.cell(200, 10, txt=f"Predicted Sales for Previous Month ({previous_month}):", ln=True)
        
        # Table Header for Previous Month Predicted Sales
        pdf.set_font('Arial', 'B', 9)
        for header in headers:
            pdf.cell(col_width, 10, header, border=1)
        pdf.ln(10)
        
        # Table contents for Previous Month Predicted Sales
        pdf.set_font('Arial', '', 9)
        for i, col in enumerate(products):
            pdf.cell(col_width, 10, str(previous_predicted_values[i]), border=1)
        pdf.ln(10)
    
    next_predicted_values = perform_prediction(next_month)
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Predicted Sales for Next Month ({next_month}):", ln=True)
    
    # Table Header for Next Month Predicted Sales
    pdf.set_font('Arial', 'B', 9)
    for header in headers:
        pdf.cell(col_width, 10, header, border=1)
    pdf.ln(10)
    
    # Table contents for Next Month Predicted Sales
    pdf.set_font('Arial', '', 9)
    for i, col in enumerate(products):
        pdf.cell(col_width, 10, str(next_predicted_values[i]), border=1)
    pdf.ln(10)
    
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
        st.session_state.predicted_values = perform_prediction(month)
        st.session_state.predicted_month = month

    if 'predicted_values' in st.session_state:
        predicted_values = st.session_state.predicted_values
        month = st.session_state.predicted_month
        fig = plot_bar_chart(df)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display the predicted sales data in a bar chart
        predicted_df = pd.DataFrame({
            'Product': ['Shumai 10 Pcs', 'Shumai 20 Pcs', 'Shumai 30 Pcs', 'Chicken Lumpia 10 Pcs'],
            'Predicted Sales': predicted_values
        })
        predicted_fig = px.bar(predicted_df, x='Product', y='Predicted Sales', color='Product', text='Predicted Sales', title='Predicted Sales Data')
        predicted_fig.update_layout(width=600, height=400)
        st.plotly_chart(predicted_fig, use_container_width=True)
        st.markdown(f"**Predicted Sales for Month {month}:**")
        prediction_df = pd.DataFrame({
            'Product': ['Shumai 10 Pcs', 'Shumai 20 Pcs', 'Shumai 30 Pcs', 'Chicken Lumpia 10 Pcs'],
            'Predicted Sales': predicted_values
        })
        st.markdown(prediction_df.style.hide(axis="index").to_html(), unsafe_allow_html=True)
        
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
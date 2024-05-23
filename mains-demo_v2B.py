import streamlit as st
import pickle
import pandas as pd
import plotly.express as px
from math import ceil
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
import tempfile
import plotly.io as pio

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

# Define a function to perform prediction
def perform_prediction(month):
    prediction = model.predict([[month]])
    return [ceil(value) for value in prediction[0]]

# Define a function to generate a PDF report using ReportLab
def generate_pdf_report(df, predicted_values, month):
    # Generate and save sales data chart
    fig = plot_bar_chart(df)
    temp_file_sales_chart = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    fig.write_image(temp_file_sales_chart.name)

    # Generate and save predicted sales data chart
    predicted_df = pd.DataFrame({
        'Product': ['Shumai 10 Pcs', 'Shumai 20 Pcs', 'Shumai 30 Pcs', 'Chicken Lumpia 10 Pcs'],
        'Predicted Sales': predicted_values
    })
    predicted_fig = px.bar(predicted_df, x='Product', y='Predicted Sales', color='Product', text='Predicted Sales', title='Predicted Sales Data')
    temp_file_predicted_chart = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    predicted_fig.write_image(temp_file_predicted_chart.name)

    # Create PDF
    pdf_path = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf').name
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    elements = []

    styles = getSampleStyleSheet()
    title_style = styles['Title']
    normal_style = styles['Normal']

    elements.append(Paragraph("Tjoean's Sales Prediction Report", title_style))
    elements.append(Spacer(1, 12))
    
    elements.append(Paragraph("Sales Data:", normal_style))
    elements.append(Spacer(1, 12))
    sales_data = [['Month', 'Shumai 10 Pcs', 'Shumai 20 Pcs', 'Shumai 30 Pcs', 'Chicken Lumpia 10 Pcs']]
    for i in range(len(df)):
        row = df.iloc[i]
        sales_data.append([row['Month'], row['Shumai 10 Pcs'], row['Shumai 20 Pcs'], row['Shumai 30 Pcs'], row['Chicken Lumpia 10 Pcs']])
    
    sales_table = Table(sales_data)
    sales_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    elements.append(sales_table)
    elements.append(Spacer(1, 12))
    
    elements.append(Paragraph(f"Predicted Sales for Month {month}:", normal_style))
    elements.append(Spacer(1, 12))
    predicted_data = [['Product', 'Predicted Sales']]
    for i, col in enumerate(['Shumai 10 Pcs', 'Shumai 20 Pcs', 'Shumai 30 Pcs', 'Chicken Lumpia 10 Pcs']):
        predicted_data.append([col, predicted_values[i]])
    
    predicted_table = Table(predicted_data)
    predicted_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    elements.append(predicted_table)
    elements.append(Spacer(1, 12))
    
    # Add previous and next month predictions
    previous_month = month - 1
    next_month = month + 1
    
    if previous_month > 0:
        elements.append(Paragraph(f"Predicted Sales for Previous Month ({previous_month}):", normal_style))
        elements.append(Spacer(1, 12))
        previous_predicted_values = perform_prediction(previous_month)
        previous_data = [['Product', 'Predicted Sales']]
        for i, col in enumerate(['Shumai 10 Pcs', 'Shumai 20 Pcs', 'Shumai 30 Pcs', 'Chicken Lumpia 10 Pcs']):
            previous_data.append([col, previous_predicted_values[i]])
        
        previous_table = Table(previous_data)
        previous_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        elements.append(previous_table)
        elements.append(Spacer(1, 12))
    
    elements.append(Paragraph(f"Predicted Sales for Next Month ({next_month}):", normal_style))
    elements.append(Spacer(1, 12))
    next_predicted_values = perform_prediction(next_month)
    next_data = [['Product', 'Predicted Sales']]
    for i, col in enumerate(['Shumai 10 Pcs', 'Shumai 20 Pcs', 'Shumai 30 Pcs', 'Chicken Lumpia 10 Pcs']):
        next_data.append([col, next_predicted_values[i]])
    
    next_table = Table(next_data)
    next_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    elements.append(next_table)
    elements.append(Spacer(1, 12))
    
    # Add charts to the PDF
    elements.append(Paragraph("Monthly Sales Chart:", normal_style))
    elements.append(Spacer(1, 12))
    elements.append(Image(temp_file_sales_chart.name))
    elements.append(Spacer(1, 12))
    
    elements.append(Paragraph("Predicted Sales Chart:", normal_style))
    elements.append(Spacer(1, 12))
    elements.append(Image(temp_file_predicted_chart.name))
    elements.append(Spacer(1, 12))

    doc.build(elements)

    return pdf_path

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
        st.subheader("Predicted Sales Data")
        predicted_df = pd.DataFrame({
            'Product': ['Shumai 10 Pcs', 'Shumai 20 Pcs', 'Shumai 30 Pcs', 'Chicken Lumpia 10 Pcs'],
            'Predicted Sales': predicted_values
        })
        predicted_fig = px.bar(predicted_df, x='Product', y='Predicted Sales', color='Product', text='Predicted Sales', title='Predicted Sales Data')
        predicted_fig.update_layout(width=600, height=400)
        st.plotly_chart(predicted_fig, use_container_width=True)
        st.subheader(f"Predicted Sales for Month {month}:")
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

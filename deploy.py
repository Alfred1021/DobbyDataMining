import pandas as pd
import streamlit as st
import plotly.express as px
from PIL import Image

st.set_page_config(page_title = "Data Mining Assignment")
st.header("Dobby Data for Data Mining")
st.subheader("Combined Data")


df = pd.read_csv("Combined.csv")
st.dataframe(df)

@st.experimental_memo
def convert_df(df):
   return df.to_csv(index=False).encode('utf-8')


csv = convert_df(df)

st.download_button(
   "Press to Download",
   csv,
   "Combine_Dobby.csv",
   "text/csv",
   key='download-csv'
)

image = Image.open('images/what people go more.png')
st.image(image, caption = "What race goes to dobby?")

image = Image.open('images/time to people.png')
st.image(image, caption = "What time does people choose to go lobby?")

image = Image.open('images/weatherofpplgo.png')
st.image(image, caption = "What weather does people choose to go to dobby?")

st.subheader("Association Rule Mining")

image = Image.open('images/chai1.png')
st.image(image, caption = "K means graph")


image = Image.open('images/chai2.png')
st.image(image, caption = "Scatter Plot of Age to Time Spent in Dobby")


st.subheader("Features Selection")

image = Image.open('images/corr.png')
st.image(image, caption = "Correlation Heatmap")

image = Image.open('images/rfcb10.png')
st.image(image, caption = "RFE Bottom 10 Features")

st.subheader("Regression Models Comparison")

image = Image.open('images/r2trend.png')
st.image(image, caption = "R2 Trend for different regression model")

image = Image.open('images/msetrend.png')
st.image(image, caption = "MSE Trend for different regression model")

st.subheader("Classification")

image = Image.open('images/ROC_NB.png')
st.image(image, caption = "Naive Bayes Classifier ROC Graph")

image = Image.open('images/NB_Precision.png')
st.image(image, caption = "Naive Bayes Classifier Precision-Recall Graph")

image = Image.open('images/ROC_DT.png')
st.image(image, caption = "Decision Tree Classifier ROC Graph")

image = Image.open('images/DT_Precision.png')
st.image(image, caption = "Decision Tree Classifier Precision-Recall Graph")

image = Image.open('images/CompareROC.png')
st.image(image, caption = "Decision Tree & Naive Bayes ROC Comparison")

image = Image.open('images/ComparePrecision.png')
st.image(image, caption = "Decision Tree & Naive Bayes Precision-Recall Comparison ")


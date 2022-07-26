import streamlit as st
import pandas as pd
import numpy as np

from wordcloud import WordCloud
import matplotlib.pyplot as plt

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


import warnings
warnings.filterwarnings('ignore')

st.title('The LAPPS Food Recommendation')



def main():
  # st.title('Explore a dataset')
  # st.write('A general purpose data exploration app')
  # file = st.sidebar.file_uploader("Upload file", type=['csv', 
  #                                              'xlsx', 
  #                                              'pickle'])
  # if not file:
  #   st.sidebar.write("Upload a .csv or .xlsx file to get started")
  #   return
  # df = get_df(file)
  # task = st.sidebar.radio('Task', ['Explore', 'Transform'], 0)
  # if task == 'Explore':
  #   explore(df)
  # else:
  #   dd=transform(df)
  #   # st.write(dd)
  pass


# FILE_ADDRESS = st.sidebar.file_uploader('Upload file')
# This variable takes the filepath after a GUI window allows you to select files from a file explorer.

# st.sidebar.image("ISB_logo.png", use_column_width=True)

st.sidebar.title('Food Recommendation')

age = st.sidebar.slider("Select yor age", 3, 100)

gender =  st.sidebar.radio(
        "Select Gender",
        ("Male", "Female")
    )

weight = st.sidebar.slider("Select weight", 10, 120)

height = st.sidebar.slider("Select Height in meters", 0.1, 2.5)

add_selectbox = st.sidebar.selectbox(
    "What's your fitness goal?",
    ("Gain Weight", "Loose Weight", "Healthy")
)


with st.sidebar:
    add_radio = st.radio(
        "Choose food preference",
        ("VEG", "NON-VEG")
    )

if height and weight is not None:
	bmi = weight/height**2

	if bmi is not None:
		st.write("BMI",(bmi))



	#Calculating BMR using using the Harris-Benedict Equation:
	BMR = 0
	if gender == 'female':
	    BMR = 655 + (9.6*weight) + (1.8*height) - (4.7*age)
	else:
	    BMR = 66 + (13.7*weight) + (5*height) - (6.8*age)

	st.write("BMR",BMR)


col1, col2, col3 = st.columns(3)

with col1:
    st.header("Butter Chicken")
    st.image("bc.jpg")

with col2:
    st.header("Russian Salad")
    st.image("rs.jpg")

with col3:
    st.header("Death by Chocolate")
    st.image("dbc.jpg")
# This is the title for the sidebar of the webpage, and stays static, based on current settings. 
# The column functionality which has been commented out further on allows the title of the main page to be dynamic.

main()

import streamlit as st
import pandas as pd
import numpy as np
import random
from wordcloud import WordCloud
import matplotlib.pyplot as plt

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


from fitness_tools.meals.meal_maker import MakeMeal

import warnings
warnings.filterwarnings('ignore')

import streamlit.components.v1 as components


st.title('The **Belly** rules the mind 😄')






from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
def create_similarity_matrix(new_description, overall_descriptions):
#Append the new description to the overall set.
    overall_descriptions.append(new_description)
    # Define a tfidf vectorizer and remove all stopwords.
    tfidf = TfidfVectorizer(stop_words="english")
    #Convert tfidf matrix by fitting and transforming the data.
    tfidf_matrix = tfidf.fit_transform(overall_descriptions)
    # output the shape of the matrix.
    tfidf_matrix.shape
    # calculating the cosine similarity matrix.
    cosine_sim = linear_kernel(tfidf_matrix,tfidf_matrix)
    return cosine_sim


def get_recommendations(new_description,overall_descriptions):
    # create the similarity matrix
    cosine_sim = create_similarity_matrix(new_description,overall_descriptions)
    # Get pairwise similarity scores of all the students with new student.
    sim_scores = list(enumerate(cosine_sim[-1]))
    # Sort the descriptions based on similarity score.
    sim_scores = sorted(sim_scores,key =lambda x:x[1],reverse= True )
    # Get the scores of top 10 descriptions.
    sim_scores = sim_scores[1:30]
    sim_scores = random.sample(sim_scores, 10)
    # Get the student indices.
    indices = [i[0]for i in sim_scores]
    
    return food_ind.iloc[indices]


def main():

  pass


# FILE_ADDRESS = st.sidebar.file_uploader('Upload file')
# This variable takes the filepath after a GUI window allows you to select files from a file explorer.

# st.sidebar.image("ISB_logo.png", use_column_width=True)

st.sidebar.title('Food Recommendation')

age = st.sidebar.slider("Select yor age", min_value=10, max_value=100, value=25)
# age = st.sidebar.slider("Select yor age", 12, 100)

gender =  st.sidebar.radio(
        "Select Gender",
        ("Male", "Female")
    )

weight = st.sidebar.slider("Select weight", 10, 120, 65)

height = st.sidebar.slider("Select Height in meters", 0.1, 2.5, 1.68)

add_selectbox = st.sidebar.selectbox(
    "What's your fitness goal?",
    ("weight_gain", "weight_loss", "maintenance")
)


with st.sidebar:
    add_radio = st.radio(
        "Choose food preference",
        ("Vegetarian", "Non-Vegetarian")
    )

if height and weight is not None:
	bmi = weight/height**2

	if bmi is not None:
		pass
		# st.write("BMI",(bmi))



	#Calculating BMR using using the Harris-Benedict Equation:
	BMR = 0
	if gender == 'female':
	    BMR = 655 + (9.6*weight) + (1.8*height) - (4.7*age)
	else:
	    BMR = 66 + (13.7*weight) + (5*height) - (6.8*age)

	# st.write("BMR",BMR)


# html_string = ""
# st.components.v1.iframe(src, width=None, height=None, scrolling=False)
# st.markdown(html_string, unsafe_allow_html=True)
height_cm = height*100

obj = MakeMeal(weight, goal='weight_gain', activity_level='moderate',
               body_type='mesomorph')
  
# Call required method
# print(obj.daily_requirements())

#calories
max_cal = obj.daily_max_calories()
min_cal = obj.daily_min_calories()
avg_cal = (max_cal + min_cal)/2


max_pro = obj.daily_max_protein()
min_pro = obj.daily_min_protein()
avg_pro = (max_pro + min_pro)/2


max_fat = obj.daily_max_fat()
min_fat = obj.daily_min_fat()
avg_fat = (max_fat + min_fat)/2


max_carb = obj.daily_max_carbs()
min_carb = obj.daily_min_carbs()
avg_carb = (max_carb + min_carb)/2


st.metric("Calories",avg_cal, 'kCal')

pro1, carb2, fat3 = st.columns(3)
pro1.metric("Protien", avg_pro, "10%")
carb2.metric("Carbs", avg_carb, "-8%")
fat3.metric("Fats", avg_fat, "4%")


food_ind = pd.read_csv('indian.csv')
# print(food_ind.head(4))

# print(add_radio) 

food_ind = food_ind.drop(['Unnamed: 0'], axis=1)

if add_radio == 'Vegetarian':
	df_food_v = food_ind[food_ind['Meal_Type'] == 'Vegetarian']
	descriptions1 = df_food_v['Calories'].apply(str) + ' ' + df_food_v['Fats'].apply(str) + ' ' + df_food_v['Protien'].apply(str) + ' ' + df_food_v['Carbohydrates'].apply(str) + ' ' +df_food_v['Meal_Type']

else:
	df_food = food_ind
	descriptions1 = df_food['Calories'].apply(str) + ' ' + df_food['Fats'].apply(str) + ' ' + df_food['Protien'].apply(str) + ' ' + df_food['Carbohydrates'].apply(str) + ' ' +df_food['Meal_Type']




st.subheader("LAPPS Recommended food")

input_desc = str(avg_cal) +' ' + str(avg_fat) + ' ' + str(avg_pro) + ' ' + str(avg_carb) + ' ' + str(add_radio)

new_description = pd.Series(input_desc)
outt = get_recommendations(new_description,descriptions1)
# HtmlFile = open("lapps_food.html", 'r', encoding='utf-8')
# source_code = HtmlFile.read() 
# # print(source_code)
# components.html(source_code)

outtt = outt[["Food_Name", "Calories","Carbohydrates","Protien","Fats","Meal_Type"]]


st.write(outtt)


# col1, col2, col3 = st.columns(3)

# with col1:
#     st.header("Butter Chicken")
#     st.image("bc.jpg")

# with col2:
#     st.header("Russian Salad")
#     st.image("rs.jpg")

# with col3:
#     st.header("Death by Chocolate")
#     st.image("dbc.jpg")
# This is the title for the sidebar of the webpage, and stays static, based on current settings. 
# The column functionality which has been commented out further on allows the title of the main page to be dynamic.

main()

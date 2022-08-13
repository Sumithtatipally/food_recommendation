import streamlit as st
import pandas as pd
import numpy as np
import random
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from datetime import datetime
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


from fitness_tools.meals.meal_maker import MakeMeal

import warnings
warnings.filterwarnings('ignore')

import streamlit.components.v1 as components



# from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode

import psycopg2

conn = psycopg2.connect(database="food_recomm",
                        host="ec2-3-6-116-139.ap-south-1.compute.amazonaws.com",
                        user="postgres",
                        password="mynewpassword",
                        port="5432")

cursor = conn.cursor()

# cursor.execute("SELECT * FROM food_data_csv")   

# cursor.commit()

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
    
    return df_food_v1.iloc[indices]


st.sidebar.title('Your Profile')


# age = st.sidebar.number_input('Enter your Age',  min_value=15, max_value=80, value=25, step=1)
your_email = st.sidebar.text_input("Enter your Email", max_chars=30)

age = st.sidebar.slider("Select Age", min_value=15, max_value=80, value=25, step=1)


gender =  st.sidebar.radio(
        "Select Gender",
        ("Male", "Female")
    )

weight = st.sidebar.slider("Select weight", 10, 140, 65)

height = st.sidebar.slider("Select Height in centimeters", min_value=135, max_value=215, value=155, step=1)

add_selectbox = st.sidebar.selectbox(
    "What's your fitness goal?",
    ("weight_gain", "weight_loss", "maintenance")
)

with st.sidebar:
    add_radio = st.radio(
        "Choose food preference",
        ("Vegetarian", "Non-Vegetarian")
    )

course_selectbox = st.sidebar.selectbox(
    "Select Course",
    ("Breakfast", "Meal")
)


cuisine_selectbox = st.sidebar.multiselect(
	     'Select Cuisine',
	     ["North Indian", "South Indian", "Asian", "Mexican", "Italian", "Others"],
	     default=["North Indian"])
# def main():

#   pass


# def get_calories_macros():

obj = MakeMeal(weight, goal=add_selectbox, activity_level='moderate',
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

	# return max_cal, max_pro, max_carb, max_fat, 


# st.metric("Calories",avg_cal, 'kCal')

pro1, carb2, fat3, kcal = st.columns(4)
kcal.metric("Calories",max_cal, 'kCal')
pro1.metric("Protien", max_pro, "10%")
carb2.metric("Carbs", max_carb, "-8%")
fat3.metric("Fats", max_fat, "4%")

# pro11, carb21, fat31, kcal1 = st.columns(4)
# kcal.metric("Remaining Calories",avg_cal, 'kCal')
# pro1.metric("Remaining Protien", avg_pro, "10%")
# carb2.metric("Remaining Carbs", avg_carb, "-8%")
# fat3.metric("Remaining Fats", avg_fat, "4%")


food_ind = pd.read_csv('indian.csv')


# food_ind_u = pd.read_csv('Food_Edited.csv')
food_ind_u = pd.read_csv('Food_final.csv')

#food preference veg/non-veg

if add_radio == 'Vegetarian':
	df_food_v = food_ind_u[food_ind_u['Meal_Type'] == 'Vegetarian']
else:
	df_food_v = food_ind_u


#course selection

if course_selectbox == "Breakfast":
	df_food_v1 = df_food_v[df_food_v['Course'] == 'Breakfast']
	descriptions1 = df_food_v1['Calories'].apply(str) + ' ' + df_food_v1['Fats'].apply(str) + ' ' + df_food_v1['Protien'].apply(str) + ' ' + df_food_v1['Carbohydrates'].apply(str) + ' ' +df_food_v1['Meal_Type'] 

else:
	df_food_v1 = df_food_v[df_food_v['Course'] == 'Meal']
	descriptions1 = df_food_v1['Calories'].apply(str) + ' ' + df_food_v1['Fats'].apply(str) + ' ' + df_food_v1['Protien'].apply(str) + ' ' + df_food_v1['Carbohydrates'].apply(str) + ' ' +df_food_v1['Meal_Type'] 



# st.subheader("LAPPS Recommended food")

input_desc = str(avg_cal/2) +' ' + str(avg_fat/2) + ' ' + str(avg_pro/2) + ' ' + str(avg_carb/2) + ' ' + str(add_radio) 


def add_data_recomm(weight, user_satisfaction, status,  height, gender, food_recom_id, food_pref, fitness_goal, cuisine, course, age, created_date, food_ids):
    # c.execute('INSERT INTO feedback (date_submitted,Q1, Q2, Q3, Q4, Q5, Q6, Q7, Q8) VALUES (?,?,?,?,?,?,?,?,?)',(date_submitted,Q1, Q2, Q3, Q4, Q5, Q6, Q7, Q8))
    cursor.execute('INSERT INTO user_data_details(weight, user_satisfaction, status,  height, gender, food_recom_id, food_pref, fitness_goal, cuisine, course, age, created_date, food_ids) VALUES (?, ?, ?,  ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',(weight, user_satisfaction, status, height, gender, food_recom_id, food_pref, fitness_goal, cuisine, course, age, created_date, food_ids))
    conn.commit()



if st.button('Recommend Food 😋',):
	new_description = pd.Series(input_desc)
	outt = get_recommendations(new_description,descriptions1)
	outtt = outt[["Food_Name", "Calories","Carbohydrates","Protien","Fats","Course","Meal_Type"]]


	st.write(outtt)
	st.write('Are you satisfied with Recommendations?')
	col1, col2= st.columns([1,1])



	food_ids = outtt['Food_Name'].tolist()
	now = datetime.now()

	# print("now =", now)

	# dd/mm/YY H:M:S
	dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
	# print("date and time =", dt_string)	
	food_recom_id = 111
	status = 1
	with col1:
		st.button("Happy 😊")
		user_satisfaction = 'yes'
		postgres_insert_query = """ INSERT INTO user_data (weight, user_satisfaction, status,  height, gender, food_recom_id, food_pref, fitness_goal, cuisine, course, age, created_date, food_ids, email) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"""
		record_to_insert = (weight, user_satisfaction, status,  height, gender, food_recom_id, add_radio, add_selectbox, cuisine_selectbox, course_selectbox, age, dt_string, food_ids, your_email)
		cursor.execute(postgres_insert_query, record_to_insert)
		conn.commit()
		# add_data_recomm( weight, user_satisfaction, status,  height, gender, food_recom_id, add_radio, add_selectbox, cuisine_selectbox, course_selectbox, age, dt_string, food_ids)
		# st.success("Feedback submitted")
	with col2:
		if st.button("Sad 🙁"):
			user_satisfaction = 'no'
			postgres_insert_query = """ INSERT INTO user_data (weight, user_satisfaction, status,  height, gender, food_recom_id, food_pref, fitness_goal, cuisine, course, age, created_date, food_ids, email) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"""
			record_to_insert = (weight, user_satisfaction, status,  height, gender, food_recom_id, add_radio, add_selectbox, cuisine_selectbox, course_selectbox, age, dt_string, food_ids,your_email)
			cursor.execute(postgres_insert_query, record_to_insert)
			conn.commit()


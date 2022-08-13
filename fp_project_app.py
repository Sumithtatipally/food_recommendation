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

st.subheader('Based on your inputs the daily ideal calories should be')



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
	     ["North Indian", "South Indian", "Asian", "Mexican", "Italian","Continental", "Others"],
	     default=["North Indian"])

lifestyle_selectbox = st.sidebar.selectbox(
	     'Select Lifestyle',
	     ["Sedentary", "Moderately Active ", "Active"]
	    )



BMI = 0
BMI = weight/height**2


calories = 0
if gender == 'Female':
    weight = float(weight)
    height = float(height)
    age = float(age)
    bmr = 655 + (9.6*weight) + (1.8*height) - (4.7*age)
    if lifestyle_selectbox == 'Sedentary':
        tdee = bmr*1.2
        if add_selectbox == 'weight_loss':
            if BMI >= 30:
                calories = tdee*0.72
            elif BMI > 25 and BMI <= 29.9:
                calories = tdee*0.77
            else:
                calories = tdee*0.8
        elif add_selectbox == 'maintain_weight':
            calories = tdee+220
        else:
            calories = tdee
    elif lifestyle_selectbox == 'Moderately Active':
        tdee = bmr*1.5
        if add_selectbox == 'weight_loss':
            if BMI >= 30:
                calories = tdee*0.72
            elif BMI > 25 and BMI <= 29.9:
                calories = tdee*0.77
            else:
                calories = tdee*0.8
        elif add_selectbox == 'maintain_weight':
            calories = tdee+220
        else:
            calories = tdee
    else:
        tdee = bmr*1.8
        if add_selectbox == 'weight_loss':
            if BMI >= 30:
                calories = tdee*0.72
            elif BMI > 25 and BMI <= 29.9:
                calories = tdee*0.77
            else:
                calories = tdee*0.8
        elif add_selectbox == 'maintain_weight':
            calories = tdee+220
        else:
            calories = tdee
else:
    weight = float(weight)
    height = float(height)
    age = float(age)
    bmr = 66 + (13.7*weight) + (5*height) - (6.8*age)
    if lifestyle_selectbox == 'Sedentary':
        tdee = bmr*1.2
        if add_selectbox == 'weight_loss':
            if BMI >= 30:
                calories = tdee*0.72
            elif BMI > 25 and BMI <= 29.9:
                calories = tdee*0.77
            else:
                calories = tdee*0.8
        elif add_selectbox == 'maintain_weight':
            calories = tdee+220
        else:
            calories = tdee
    elif lifestyle_selectbox == 'Moderately Active':
        tdee = bmr*1.5
        if add_selectbox == 'weight_loss':
            if BMI >= 30:
                calories = tdee*0.72
            elif BMI > 25 and BMI <= 29.9:
                calories = tdee*0.77
            else:
                calories = tdee*0.8
        elif add_selectbox == 'maintain_weight':
            calories = tdee+220
        else:
            calories = tdee
    else:
        tdee = bmr*1.8
        if add_selectbox == 'weight_loss':
            if BMI >= 30:
                calories = tdee*0.72
            elif BMI > 25 and BMI <= 29.9:
                calories = tdee*0.77
            else:
                calories = tdee*0.8
        elif add_selectbox == 'maintain_weight':
            calories = tdee+220
        else:
            calories = tdee



print(int(calories))

calories = int(calories)
pro = int(calories*0.3)
fat = int(calories*0.2)
carbs = int(calories*0.5)

print(pro, fat, carbs)


calori,pr, car, fa  = st.columns(4)
calori.metric("Calories",calories, 'kCal')
pr.metric("Protien", pro, "gm")
car.metric("Carbs", carbs, "gm")
fa.metric("Fats", fat, "gm")

# pro11, carb21, fat31, kcal1 = st.columns(4)
# kcal.metric("Remaining Calories",avg_cal, 'kCal')
# pro1.metric("Remaining Protien", avg_pro, "10%")
# carb2.metric("Remaining Carbs", avg_carb, "-8%")
# fat3.metric("Remaining Fats", avg_fat, "4%")


food_ind = pd.read_csv('indian.csv')


# food_ind_u = pd.read_csv('Food_Edited.csv')
# food_ind_u = pd.read_csv('Food_final.csv')
sql_query = pd.read_sql_query ('''
                               SELECT * FROM food_data_csv
                               ''', conn)

food_ind_u = pd.DataFrame(sql_query)
#food preference veg/non-veg

if add_radio == 'Vegetarian':
	df_food_v = food_ind_u[food_ind_u['meal_type'] == 'Vegetarian']
else:
	df_food_v = food_ind_u


#course selection

if course_selectbox == "Breakfast":
	df_food_v1 = df_food_v[df_food_v['course'] == 'Breakfast']
	descriptions1 = df_food_v1['calories'].apply(str) + ' ' + df_food_v1['fats'].apply(str) + ' ' + df_food_v1['protien'].apply(str) + ' ' + df_food_v1['carbohydrates'].apply(str) + ' ' +df_food_v1['meal_type'] 

else:
	df_food_v1 = df_food_v[df_food_v['course'] == 'Meal']
	descriptions1 = df_food_v1['calories'].apply(str) + ' ' + df_food_v1['fats'].apply(str) + ' ' + df_food_v1['protien'].apply(str) + ' ' + df_food_v1['carbohydrates'].apply(str) + ' ' +df_food_v1['meal_type'] 



# st.subheader("LAPPS Recommended food")

input_desc = str(calories) +' ' + str(fat) + ' ' + str(pro) + ' ' + str(carbs) + ' ' + str(add_radio) 


def add_data_recomm(weight, user_satisfaction, status,  height, gender, food_recom_id, food_pref, fitness_goal, cuisine, course, age, created_date, food_ids):
    # c.execute('INSERT INTO feedback (date_submitted,Q1, Q2, Q3, Q4, Q5, Q6, Q7, Q8) VALUES (?,?,?,?,?,?,?,?,?)',(date_submitted,Q1, Q2, Q3, Q4, Q5, Q6, Q7, Q8))
    cursor.execute('INSERT INTO user_data_details(weight, user_satisfaction, status,  height, gender, food_recom_id, food_pref, fitness_goal, cuisine, course, age, created_date, food_ids) VALUES (?, ?, ?,  ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',(weight, user_satisfaction, status, height, gender, food_recom_id, food_pref, fitness_goal, cuisine, course, age, created_date, food_ids))
    conn.commit()





if st.button('Recommend Food 😋',):
	new_description = pd.Series(input_desc)
	outt = get_recommendations(new_description,descriptions1)
	outtt = outt[["food_name", "calories","carbohydrates","protien","fats","course","cuisine","meal_type"]]


	st.write(outtt)
	st.write('Are you satisfied with Recommendations?')
	col1, col2= st.columns([1,1])



	food_ids = outtt['food_name'].tolist()
	now = datetime.now()

	# print("now =", now)

	# dd/mm/YY H:M:S
	dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
	# print("date and time =", dt_string)	
	food_recom_id = 111
	status = 1

	def run_query_yes():
		user_satisfaction = 'yes'
		postgres_insert_query = """ INSERT INTO user_data (weight, user_satisfaction, status,  height, gender, food_recom_id, food_pref, fitness_goal, cuisine, course, age, created_date, food_ids, email, lifestyle) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"""
		record_to_insert = (weight, user_satisfaction, status,  height, gender, food_recom_id, add_radio, add_selectbox, cuisine_selectbox, course_selectbox, age, dt_string, food_ids, your_email, lifestyle_selectbox)
		cursor.execute(postgres_insert_query, record_to_insert)
		conn.commit()
		conn.close()
		st.success('Thanks for Feedback.')
		return()


	def run_query_no():
		user_satisfaction = 'no'
		postgres_insert_query = """ INSERT INTO user_data (weight, user_satisfaction, status,  height, gender, food_recom_id, food_pref, fitness_goal, cuisine, course, age, created_date, food_ids, email, lifestyle) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"""
		record_to_insert = (weight, user_satisfaction, status,  height, gender, food_recom_id, add_radio, add_selectbox, cuisine_selectbox, course_selectbox, age, dt_string, food_ids, your_email, lifestyle_selectbox)
		cursor.execute(postgres_insert_query, record_to_insert)
		conn.commit()
		conn.close()
		st.success('Thanks for Feedback.')
		return()


	with col1:
			# st.button("Happy 😊")
		if st.button("Happy 😊",on_click = run_query_yes):
			run_query_yes()
		
	with col2:
		if st.button("Sad 🙁",on_click = run_query_no):
			run_query_no()







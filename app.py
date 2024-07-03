import streamlit as st

import pickle 
import tensorflow
from tensorflow.keras.models import load_model
import pandas as pd

## loading data
df = pd.read_csv('data/online_course_engagement_data.csv')

## loading preprocessor & ANN model
preprocessor = pickle.load(open('preprocessor.pkl','rb'))
model = load_model('model.h5')

st.title("Welcome to the home page")

st.sidebar.title("Choose the Inputs")
CourseCategory = st.sidebar.selectbox("select Course Category",options=df['CourseCategory'].unique())
TimeSpentOnCourse = st.sidebar.slider("select Time Spent On Course",min_value=df['TimeSpentOnCourse'].min(),max_value=df['TimeSpentOnCourse'].max())
NumberOfVideosWatched = st.sidebar.slider("select Number Of Videos Watched",min_value=df['NumberOfVideosWatched'].min(),max_value=df['NumberOfVideosWatched'].max())
NumberOfQuizzesTaken = st.sidebar.slider("select Number Of Quizzes Taken",min_value=df['NumberOfQuizzesTaken'].min(),max_value=df['NumberOfQuizzesTaken'].max())
QuizScores = st.sidebar.slider("select Quiz Scores",min_value=df['QuizScores'].min(),max_value=df['QuizScores'].max())
CompletionRate = st.sidebar.slider("select CompletionRate",min_value=df['CompletionRate'].min(),max_value=df['CompletionRate'].max())
DeviceType = st.sidebar.selectbox("select Course Category",options=df['DeviceType'].unique())

# Example input data
input_data = {
    'CourseCategory': [CourseCategory],
    'TimeSpentOnCourse': [TimeSpentOnCourse],
    'NumberOfVideosWatched': [NumberOfVideosWatched],
    'NumberOfQuizzesTaken': [NumberOfQuizzesTaken],
    'QuizScores': [QuizScores],
    'CompletionRate':[CompletionRate],
    'DeviceType': [DeviceType],
}
def get_input(input_data):
    input_dataframe = pd.DataFrame(input_data)
    preprocessed_data = preprocessor.transform(input_dataframe)
    result = model.predict(preprocessed_data)

    if result[0][0] > 0.5:
        st.success('Course is Purchased', icon="✅")
    else:
        st.warning('Course is Not Purchased')


if st.button("predict"):
    get_input(input_data)
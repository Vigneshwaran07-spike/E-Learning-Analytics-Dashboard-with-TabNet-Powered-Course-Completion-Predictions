import streamlit as st
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import plotly.express as px
import joblib
from pytorch_tabnet.tab_model import TabNetClassifier

# Function to run SQL query using SQLAlchemy
def run_query(query):
    try:
        engine = create_engine("mysql+mysqlconnector://root:root@localhost/elearning_platform")
        df = pd.read_sql(query, engine)
        return df
    except Exception as err:
        st.error(f"Database Error: {err}")
        return None

# Load TabNet model, scaler, and feature columns
try:
    model = TabNetClassifier()
    model.load_model('best_model.zip')
    scaler = joblib.load('scaler.pkl')
    feature_columns = joblib.load('feature_columns.pkl')
except FileNotFoundError as e:
    st.error(f"Model or scaler file not found: {e}")
    st.stop()

# Streamlit app title
st.title("E-Learning Platform Analytics Dashboard")
st.markdown("Interactive dashboard with AI-powered predictions for course completion, enrollments, and performance metrics.")

# Sidebar for navigation
st.sidebar.header("Navigation")
section = st.sidebar.selectbox("Choose a section", [
    "Popular Courses",
    "User Engagement Trends",
    "Course Completion Rates",
    "High-Performing Courses",
    "Predict Completion Likelihood"
])

# Feature 1: Popular Courses
if section == "Popular Courses":
    st.header("Popular Courses by Enrollment")
    st.markdown("Shows courses with more than 2 enrollments.")
    query = """
    SELECT c.course_name, COUNT(*) AS enrollment_count
    FROM courses c
    JOIN enrollments e ON c.course_id = e.course_id
    GROUP BY c.course_name
    HAVING COUNT(*) > 2
    ORDER BY enrollment_count DESC;
    """
    df = run_query(query)
    if df is not None:
        st.dataframe(df)
        fig = px.bar(df, x="course_name", y="enrollment_count", title="Popular Courses")
        st.plotly_chart(fig)

# Feature 2: User Engagement Trends
elif section == "User Engagement Trends":
    st.header("Monthly Enrollment Trends")
    st.markdown("Displays enrollment counts per month in 2025.")
    query = """
    SELECT DATE_FORMAT(enrollment_date, '%Y-%m') AS month, COUNT(*) AS enrollments
    FROM enrollments
    WHERE YEAR(enrollment_date) = 2025
    GROUP BY month
    ORDER BY month;
    """
    df = run_query(query)
    if df is not None:
        st.dataframe(df)
        fig = px.line(df, x="month", y="enrollments", title="Enrollment Trends")
        st.plotly_chart(fig)

# Feature 3: Course Completion Rates
elif section == "Course Completion Rates":
    st.header("Completion Rates by Category")
    st.markdown("Shows percentage of completed enrollments per category.")
    query = """
    SELECT c.category,
           COUNT(CASE WHEN e.completion_status = 'Completed' THEN 1 END) * 100.0 / COUNT(*) AS completion_rate
    FROM courses c
    JOIN enrollments e ON c.course_id = e.course_id
    GROUP BY c.category;
    """
    df = run_query(query)
    if df is not None:
        st.dataframe(df)
        fig = px.pie(df, names="category", values="completion_rate", title="Completion Rates")
        st.plotly_chart(fig)

# Feature 4: High-Performing Courses
elif section == "High-Performing Courses":
    st.header("Courses with Above-Average Ratings")
    st.markdown("Lists courses with ratings above the platform average.")
    query = """
    SELECT course_name, rating
    FROM courses
    WHERE rating > (SELECT AVG(rating) FROM courses)
    ORDER BY rating DESC;
    """
    df = run_query(query)
    if df is not None:
        st.dataframe(df)

# Feature 5: Predict Completion Likelihood
elif section == "Predict Completion Likelihood":
    st.header("Predict Course Completion Likelihood")
    st.markdown("Uses a TabNet deep learning model to predict if a user will complete a course.")
    
    # Fetch user and course options
    users_df = run_query("SELECT user_id, name FROM users;")
    courses_df = run_query("SELECT course_id, course_name, rating, price, category FROM courses;")
    
    if users_df is not None and courses_df is not None:
        user_options = {row['name']: row['user_id'] for _, row in users_df.iterrows()}
        course_options = {row['course_name']: row for _, row in courses_df.iterrows()}
        
        # User inputs
        user_name = st.selectbox("Select User", list(user_options.keys()))
        course_name = st.selectbox("Select Course", list(course_options.keys()))
        enrollment_date = st.date_input("Enrollment Date")
        
        if st.button("Predict"):
            user_id = user_options[user_name]
            course = course_options[course_name]
            join_date = run_query(f"SELECT join_date FROM users WHERE user_id = {user_id};")['join_date'][0]
            
            # Compute features
            days_since_joined = (pd.to_datetime(enrollment_date) - pd.to_datetime(join_date)).days
            days_since_enrolled = (pd.to_datetime('2025-05-28') - pd.to_datetime(enrollment_date)).days
            user_enrollment_count = run_query(
                f"SELECT COUNT(*) AS count FROM enrollments WHERE user_id = {user_id};"
            )['count'][0]
            avg_category_rating = run_query(
                f"SELECT AVG(rating) AS avg_rating FROM courses WHERE category = '{course['category']}';"
            )['avg_rating'][0]
            course_popularity = run_query(
                f"SELECT COUNT(*) AS count FROM enrollments WHERE course_id = {course['course_id']};"
            )['count'][0]
            
            # Prepare input data with one-hot encoding for category
            category_columns = [col for col in feature_columns if col.startswith('category_')]
            input_data = pd.DataFrame([[
                course['rating'],
                course['price'],
                *[1 if col == f"category_{course['category']}" else 0 for col in category_columns],
                days_since_joined,
                days_since_enrolled,
                user_enrollment_count,
                avg_category_rating,
                course_popularity
            ]], columns=feature_columns)
            
            # Scale numerical features
            numerical_indices = [0, 1] + [feature_columns.index(col) for col in feature_columns if col in [
                'days_since_joined', 'days_since_enrolled', 'user_enrollment_count',
                'avg_category_rating', 'course_popularity'
            ]]
            input_data.iloc[:, numerical_indices] = scaler.transform(input_data.iloc[:, numerical_indices])
            
            # Predict
            try:
                prediction = model.predict_proba(input_data.values.astype(np.float64))[0][1]
                st.write(f"Predicted Completion Likelihood for {user_name} in {course_name}: {prediction:.2%}")
            except Exception as e:
                st.error(f"Prediction error: {e}")
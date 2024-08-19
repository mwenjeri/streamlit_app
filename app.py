import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import randint
import time

# App title
st.title('Employee Performance Analysis')

st.write('This project focuses on analyzing employee performance within an organization to gain insights into factors affecting performance ratings. This is done by examining a dataset that includes various employee metrics such as environment satisfaction, work-life balance, age, and department.')

# Import dataset 
df = pd.read_csv("C:/Users/mwenj/OneDrive/Desktop/streaml_app/processed.csv")

# Drop specified columns that are not needed for model training
columns_to_drop = ['EmpNumber', 'Gender', 'EmpJobSatisfaction']
df = df.drop(columns=columns_to_drop, errors='ignore')

# Display the first five rows of the dataset
st.write("### Dataset Overview")
st.write(df.head())

# Add a slider to select the number of rows to view
num_rows = st.slider(
    'Select the number of rows to view', 
    min_value=1, 
    max_value=len(df), 
    value=10, 
    step=1
)

# Display the selected number of rows from the dataframe
st.write(f"### Viewing first {num_rows} rows of the dataset")
st.write(df.head(num_rows))

# Initialize LabelEncoder
le = LabelEncoder()

# Encode categorical columns
categorical_columns = [
    'EducationBackground', 'MaritalStatus', 'EmpDepartment', 
    'EmpJobRole', 'BusinessTravelFrequency', 'OverTime', 'Attrition'
]

# Check which categorical columns exist before encoding
for col in categorical_columns:
    if col in df.columns:
        df[col] = le.fit_transform(df[col])
    else:
        st.warning(f"Column '{col}' not found in DataFrame.")

# Prepare features and target variable
X = df.drop("PerformanceRating", axis=1)
y = df["PerformanceRating"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
rf = RandomForestRegressor()

# Define the parameter distribution for RandomizedSearchCV
param_dist = {
    'n_estimators': randint(50, 200),  # Random integer between 50 and 200
    'max_features': ['sqrt', 'log2', 1.0],  # Explicitly set max_features, removing 'auto'
    'max_depth': [5, 10, 20, 30],  # Maximum depth of the tree
    'min_samples_split': randint(2, 10),  # Random integer between 2 and 10
    'min_samples_leaf': randint(1, 4),  # Random integer between 1 and 4
    'bootstrap': [True, False]  # Whether bootstrap samples are used when building trees
}

# Setup RandomizedSearchCV with a progress bar
with st.spinner("Training model..."):
    random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_dist, n_iter=5, cv=5,
                                       scoring='neg_mean_squared_error', n_jobs=-1, verbose=0, random_state=42)
    
    # Train the model
    random_search.fit(X_train, y_train)
    
    # Simulate progress bar update for demo purposes (this is optional and can be removed)
    progress_bar = st.progress(0)
    for i in range(100):
        time.sleep(0.01)  # Simulate processing time
        progress_bar.progress(i + 1)
    progress_bar.empty()  # Clear progress bar

# Display best parameters and score
st.write("### Model Training Results")
best_params = random_search.best_params_
best_score = random_search.best_score_

st.write("Best parameters:", best_params)
st.write("Best score:", best_score)

# Make predictions on the test set
y_pred = random_search.predict(X_test)

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display metrics
st.write("### Model Performance Metrics")
st.write(f"Mean Absolute Error (MAE): {mae}")
st.write(f"Mean Squared Error (MSE): {mse}")
st.write(f"R-squared (R²): {r2}")

# Define label encoders for user input
le_dict = {
    'EducationBackground': LabelEncoder().fit(['Bachelor', 'Master', 'PhD']),
    'MaritalStatus': LabelEncoder().fit(['Single', 'Married']),
    'EmpDepartment': LabelEncoder().fit(['Sales', 'Research & Development', 'Human Resources']),
    'EmpJobRole': LabelEncoder().fit(['Sales Executive', 'Research Scientist', 'Healthcare Representative']),
    'BusinessTravelFrequency': LabelEncoder().fit(['Non-Travel', 'Travel_Rarely', 'Travel_Frequently']),
    'OverTime': LabelEncoder().fit(['Yes', 'No']),
    'Attrition': LabelEncoder().fit(['Yes', 'No'])
}

# User input for new data using expanders
with st.sidebar.expander("Personal Information"):
    education_background = st.selectbox("Education Background", le_dict['EducationBackground'].classes_)
    marital_status = st.selectbox("Marital Status", le_dict['MaritalStatus'].classes_)

with st.sidebar.expander("Job Information"):
    emp_department = st.selectbox("Department", le_dict['EmpDepartment'].classes_)
    emp_job_role = st.selectbox("Job Role", le_dict['EmpJobRole'].classes_)
    business_travel = st.selectbox("Business Travel Frequency", le_dict['BusinessTravelFrequency'].classes_)
    distance_from_home = st.number_input("Distance from Home")
    emp_education_level = st.number_input("Education Level")
    env_satisfaction = st.number_input("Environment Satisfaction")
    hourly_rate = st.number_input("Hourly Rate")
    job_involvement = st.number_input("Job Involvement")
    job_level = st.number_input("Job Level")
    job_satisfaction = st.number_input("Job Satisfaction")
    num_companies_worked = st.number_input("Number of Companies Worked")
    overtime = st.selectbox("Overtime", le_dict['OverTime'].classes_)
    last_salary_hike_percent = st.number_input("Last Salary Hike Percent")

with st.sidebar.expander("Additional Information"):
    relationship_satisfaction = st.number_input("Relationship Satisfaction")
    total_work_experience = st.number_input("Total Work Experience in Years")
    training_times_last_year = st.number_input("Training Times Last Year")
    work_life_balance = st.number_input("Work Life Balance")
    experience_years_at_company = st.number_input("Experience Years at Company")
    experience_years_in_current_role = st.number_input("Experience Years in Current Role")
    years_since_last_promotion = st.number_input("Years Since Last Promotion")
    years_with_current_manager = st.number_input("Years with Current Manager")
    attrition = st.selectbox("Attrition", le_dict['Attrition'].classes_)

# Encode user input
encoded_input = [
    le_dict['EducationBackground'].transform([education_background])[0],
    le_dict['MaritalStatus'].transform([marital_status])[0],
    le_dict['EmpDepartment'].transform([emp_department])[0],
    le_dict['EmpJobRole'].transform([emp_job_role])[0],
    le_dict['BusinessTravelFrequency'].transform([business_travel])[0],
    distance_from_home,
    emp_education_level,
    env_satisfaction,
    hourly_rate,
    job_involvement,
    job_level,
    job_satisfaction,
    num_companies_worked,
    le_dict['OverTime'].transform([overtime])[0],
    last_salary_hike_percent,
    relationship_satisfaction,
    total_work_experience,
    training_times_last_year,
    work_life_balance,
    experience_years_at_company,
    experience_years_in_current_role,
    years_since_last_promotion,
    years_with_current_manager,
    le_dict['Attrition'].transform([attrition])[0]
]

# Define the expected columns from training data without 'PerformanceRating'
expected_columns = X_train.columns

# Create DataFrame for prediction without 'PerformanceRating'
df_user_input = pd.DataFrame([encoded_input], columns=expected_columns)

# Predict using the model
if st.sidebar.button('Predict'):
    try:
        st.sidebar.write('Making prediction...')
        # Simulate progress for demonstration purposes
        progress_bar = st.sidebar.progress(0)
        for i in range(100):
            time.sleep(0.01)  # Simulate processing time
            progress_bar.progress(i + 1)
        prediction = random_search.predict(df_user_input)
        progress_bar.progress(100)  # Ensure progress bar is fully filled
        st.sidebar.write('Predicted Performance Rating:', prediction[0])
    except ValueError as e:
        st.sidebar.write('Error:', str(e))

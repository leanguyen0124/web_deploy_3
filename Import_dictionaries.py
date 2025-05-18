
###Import dictionaries
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import joblib
import streamlit as st
import datetime
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

###Import data set
data_df = pd.read_csv("https://raw.githubusercontent.com/leanguyen0124/web_deploy_3/refs/heads/main/gym_members_exercise_tracking.csv")

categorical_columns = ['Gender', 'Workout_Type']
numerical_columns = ['Age', 'Weight (kg)', 'Avg_BPM', 'Session_Duration (hours)', 'Workout_Frequency (days/week)', 'Experience_Level']

X = data_df[categorical_columns + numerical_columns]
y = data_df['Calories_Burned']

preprocessor = ColumnTransformer(
    transformers=[
        ('', OneHotEncoder(), categorical_columns)
    ],
    remainder='passthrough'  # giữ lại các cột còn lại (Age)
)
data_df= preprocessor.fit_transform(data_df)

sc = StandardScaler()
X = sc.fit_transform(X)

# Pipeline xử lý + model
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_columns),
        ('num', StandardScaler(), numerical_columns)
    ]
)

pipeline = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('Scaler', sc),
    ('regressor', RandomForestRegressor())
])

# GridSearchCV
param_grid = {
    'regressor__n_estimators': [50,100,200],
    'regressor__max_depth': [ None, 10, 20]
}
grid = GridSearchCV(pipeline, param_grid, cv=5)
grid.fit(X, y)

joblib.dump(grid.best_estimator_, 'model.pkl')
model = joblib.load('model.pkl')

# File lưu trữ lịch sử
DATA_FILE = 'workout_history.csv'

# Hàm đọc dữ liệu lịch sử
def load_data():
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE, parse_dates=['date'])
    else:
        return pd.DataFrame(columns=['Date','Age', 'Gender', 'Weight (kg)', 'Avg_BPM', 'Session_Duration (hours)', 'Workout_Type', 'Workout_Frequency (days/week)',  'Experience_Level', 'Calories_Burned'])

# Hàm lưu dữ liệu mới vào file
def save_data(df):
    df.to_csv(DATA_FILE, index=False)

# Streamlit app
st.title("Dự đoán calories tiêu thụ")

# Form nhập thông tin người dùng
with st.form("input_form"):
    Age = st.number_input("Tuổi", min_value=1, max_value=120, value=25)
    Gender = st.selectbox("Giới tính", ["Female", "Male"])
    Weight = st.number_input("Cân nặng", min_value=0, max_value=200, value=70)
    Avg_BPM = st.number_input("Nhịp tim trung bình", min_value=0, max_value=200, value=80)
    Session_Duration = st.number_input("Thời gian tập luyện", min_value=0, max_value=5, value=1)
    Workout_Type = st.selectbox("HÌnh thức tập luyện", ["HIIT", "Yoga", "Cardio", "Strength"])
    Workout_Frequency = st.number_input("Tần suất tập luyện", min_value=0, max_value=7, value=3)
    Experience_Level = st.selectbox("Mức độ thuần thục", ["1", "2","3"])
    
    submitted = st.form_submit_button("Dự đoán")

if submitted:
    # Chuẩn bị dữ liệu cho mô hình
    X = pd.DataFrame([{
    'Age': Age,
    'Gender': Gender,
    'Weight (kg)': Weight,
    'Avg_BPM': Avg_BPM,
    'Session_Duration (hours)': Session_Duration,
    'Workout_Type': Workout_Type,
    'Workout_Frequency (days/week)': Workout_Frequency,
    'Experience_Level': Experience_Level
    }])
    calories_pred = model.predict(X)[0]
    
    st.success(f"Bạn đã tiêu thụ khoảng: {calories_pred:.2f} calories")

    # Lưu vào lịch sử
    data = load_data()
    new_record = {
        'Date': datetime.datetime.now(),
        'Age': Age,
        'Gender' : Gender,
        'Weight (kg)': Weight,
        'Avg_BPM': Avg_BPM,
        'Session_Duration (hours)': Session_Duration,
        'Workout_Type': Workout_Type,
        'Workout_Frequency (days/week)' : Workout_Frequency,
        'Experience_Level': Experience_Level,
        'Calories_burned': calories_pred
    }
    data = data.append(new_record, ignore_index=True)
    save_data(data)

# Hiển thị lịch sử và biểu đồ
st.header("Lịch sử tập luyện")

data = load_data()

if data.empty:
    st.info("Chưa có dữ liệu lịch sử tập luyện nào.")
else:
    st.dataframe(data[['Date','Age', 'Gender', 'Weight (kg)', 'Avg_BPM', 'Session_Duration (hours)', 'Workout_Type', 'Workout_Frequency (days/week)',  'Experience_Level', 'Calories_Burned'
]])

    # Vẽ biểu đồ calories theo thời gian
    plt.figure(figsize=(10,4))
    plt.plot(data['date'], data['calories'], marker='o')
    plt.title('Lượng calories tiêu thụ theo thời gian')
    plt.xlabel('Ngày')
    plt.ylabel('Calories')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)
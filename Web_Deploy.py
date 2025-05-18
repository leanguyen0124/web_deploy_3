import streamlit as st
import pandas as pd
import joblib
import datetime
import os
import matplotlib.pyplot as plt

# Load model
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
    X = [['Age', 'Gender', 'Weight (kg)', 'Avg_BPM', 'Session_Duration (hours)', 'Workout_Type', 'Workout_Frequency (days/week)',  'Experience_Level']]
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
#!/usr/bin/env python
# coding: utf-8

# In[1]:


###Import dictionaries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


###Import data set
data_df = pd.read_csv("https://raw.githubusercontent.com/leanguyen0124/web_deploy_3/refs/heads/main/gym_members_exercise_tracking.csv")
pd.set_option('display.max_columns', None)
print ('sample bộ data_set')
print ( data_df.head() ) ## method head() lấy giá trị của 5 dòng đầu tiên của tập dữ liệu


# In[3]:


###Data preprocessing

###Data cleaning
data_df = data_df.drop_duplicates() ##loại bỏ những data bị trùng lặp
data_df["Workout_Type"] = data_df["Workout_Type"].str.strip() ##loại bỏ những space không cần thiết 
###nếu specific hơn cho các ký tự khác thì điền ở trong dấu ngoặc, gộp lại chung thành 1 chuỗi cái ký tự để cùng đồng thời xóa được nhiều cái

##data missing 
print('N của các biến trong tập dữ liệu')
from IPython.display import display
display(data_df.info())  ##method info() giúp xác định số giá trị của các atributes trong tập dữ liệu


# In[4]:


####Data visualization


# In[5]:


print('Mô tả thống kê cho tập dữ liệu')
from IPython.display import display
display(data_df.describe())

sns.displot(data_df["Age"], color= '#67AE6E', height=5, aspect=1, binwidth=1)
plt.title("Phân bố dữ liệu về độ tuổi", fontsize=16)
plt.xlabel("Tuổi", fontsize=12)
plt.ylabel("Số lượng", fontsize=12)
plt.tight_layout()
#plt.show()


# In[6]:


bins_age = [17, 25, 35, 45, 55, 100]
labels_age = ['18-25', '26-35', '36-45', '46-55', '56+']
data_df['Age_Group'] = pd.cut(data_df['Age'], bins=bins_age, labels=labels_age)
grouped_age = data_df.groupby('Age_Group')['Calories_Burned']
mean_values_age = grouped_age.mean()
median_values_age = grouped_age.median()
plt.plot(mean_values_age.index, mean_values_age.values, marker='o', label='Mean', color='#67AE6E')
plt.plot(median_values_age.index, median_values_age.values, marker='o', label='Median', color = '#FF9B17')
plt.title("Lượng calories sử dụng theo nhóm tuổi", fontsize=16)
plt.xlabel("Nhóm tuổi", fontsize=12)
plt.legend()
plt.ylabel("Lượng calories sử dụng", fontsize=12)
plt.tight_layout()
#plt.show()


# In[7]:


dem_gender = data_df['Gender'].value_counts().sum()
dem_female = (data_df['Gender'] == 'Female').sum()
female = dem_female*100/dem_gender
label_gender = ['Nữ', 'Nam']
sizes = [female, 100-female]
plt.pie(sizes, labels=label_gender,colors=['#67AE6E','#90C67C'], autopct='%1.1f%%', startangle=90)
plt.axis('equal')
plt.title("Phân bố giới tính")
plt.tight_layout()
#plt.show()


# In[8]:


bins_calories = [300, 599, 799, 999, 1199, 1399, 1599,2000]
labels_calories = ['300-599', '600-799', '800-999', '1000-1199', '1200-1399', '1400-1599', '1600+']
data_df['Calories_Group'] = pd.cut(data_df['Calories_Burned'], bins=bins_calories, labels=labels_calories)
# Tạo bảng đếm số lượng từng giới tính trong mỗi nhóm
gender_counts = data_df.groupby(['Calories_Group', 'Gender']).size().unstack()
# Tính phần trăm theo hàng (mỗi nhóm cộng lại = 100%)
gender_percentage = gender_counts.div(gender_counts.sum(axis=1), axis=0) * 100
# Vẽ stacked bar chart
gender_percentage.plot(kind='bar', stacked=True, color=['#67AE6E','#90C67C'])
plt.title("Lượng calories sử dụng theo giới tính", fontsize=16)
plt.xlabel("Lượng calories sử dụng", fontsize=10)
plt.ylabel("Giới tính", fontsize=10)
plt.tight_layout()
#plt.show()


# In[9]:


sns.displot(data_df["Weight (kg)"], color= '#67AE6E', height=5, aspect=1, bins=15)
plt.title("Phân bố dữ liệu về cân nặng", fontsize=16,)
plt.xlabel("Cân nặng", fontsize=12)
plt.ylabel("Số lượng", fontsize=12)
plt.tight_layout()
#plt.show()


# In[10]:


bins_weight = [39, 49, 59, 69, 79, 99, 200]
labels_weight = ['40-49', '50-59', '60-69', '70-79', '80-99', '100+']
data_df['Weight_Group'] = pd.cut(data_df['Weight (kg)'], bins=bins_weight, labels=labels_weight)
grouped_weight = data_df.groupby('Weight_Group')['Calories_Burned']
mean_values_weight = grouped_weight.mean()
median_values_weight = grouped_weight.median()
plt.plot(mean_values_weight.index, mean_values_weight.values, marker='o', label='Mean', color='#67AE6E')
plt.plot(median_values_weight.index, median_values_weight.values, marker='o', label='Median', color = '#FF9B17')
plt.title("Lượng calories sử dụng theo nhóm cân nặng", fontsize=16)
plt.xlabel("Nhóm cân nặng", fontsize=12)
plt.ylabel("Lượng calories sử dụng", fontsize=12)
plt.legend()
plt.tight_layout()
#plt.show()


# In[11]:


sns.displot(data_df["Height (m)"], color= '#67AE6E', height=5, aspect=1, bins=15)
plt.title("Phân bố dữ liệu về chiều cao", fontsize=16)
plt.xlabel("Chiều cao", fontsize=12)
plt.ylabel("Số lượng", fontsize=12)
plt.tight_layout()
#plt.show()


# In[12]:


bins_height = [1.49, 1.59, 1.69, 1.79, 1.89, 2.5]
labels_height = ['1m5-1m59', '1m6-1m69', '1m7-1m79', '1m8-1m89', '1m9+']
data_df['Height_Group'] = pd.cut(data_df['Height (m)'], bins=bins_height, labels=labels_height)
grouped_height = data_df.groupby('Height_Group')['Calories_Burned']
mean_values_height = grouped_height.mean()
median_values_height = grouped_height.median()
plt.plot(mean_values_height.index, mean_values_height.values, marker='o', label='Mean', color='#67AE6E')
plt.plot(median_values_height.index, median_values_height.values, marker='o', label='Median', color = '#FF9B17')
plt.title("Lượng calories sử dụng theo nhóm chiều cao", fontsize=16)
plt.xlabel("Nhóm chiều cao", fontsize=12)
plt.ylabel("Lượng calories sử dụng", fontsize=12)
plt.legend()
plt.tight_layout()
#plt.show()


# In[13]:


sns.displot(data_df["Avg_BPM"],color= '#67AE6E', height=5, aspect=1, binwidth=5)
plt.title("Phân bố dữ liệu về BPM trung bình", fontsize=16)
plt.xlabel("BPM trung bình", fontsize=12)
plt.ylabel("Số lượng", fontsize=12)
plt.tight_layout()
#plt.show()


# In[14]:


bins_bpm = [119, 129, 139, 149, 159, 200]
labels_bpm = ['120-129', '130-139', '140-149', '150-159', '160+']
data_df['BPM_Group'] = pd.cut(data_df['Avg_BPM'], bins=bins_bpm, labels=labels_bpm)
grouped_bpm = data_df.groupby('BPM_Group')['Calories_Burned']
mean_values_bpm = grouped_bpm.mean()
median_values_bpm = grouped_bpm.median()
plt.plot(mean_values_bpm.index, mean_values_bpm.values, marker='o', label='Mean', color='#67AE6E')
plt.plot(median_values_bpm.index, median_values_bpm.values, marker='o', label='Median', color = '#FF9B17')
plt.title("Lượng calories sử dụng theo BPM trung bình", fontsize=16)
plt.xlabel("BPM trung bình", fontsize=12)
plt.ylabel("Lượng calories sử dụng", fontsize=12)
plt.legend()
plt.tight_layout()
#plt.show()


# In[15]:


sns.displot(data_df["Session_Duration (hours)"],color= '#67AE6E', height=5, aspect=1, bins=15)
plt.title("Phân bố dữ liệu về thời gian tập luyện", fontsize=16)
plt.xlabel("Thời gian tập luyện", fontsize=12)
plt.ylabel("Số lượng", fontsize=12)
plt.tight_layout()
#plt.show()


# In[16]:


bins_hour = [0.49, 0.99, 1.49, 3]
labels_hour = ['0.5-1h', '1h-1.5h', '1.5h+']
data_df['Hour_Group'] = pd.cut(data_df['Session_Duration (hours)'], bins=bins_hour, labels=labels_hour)
grouped_hour = data_df.groupby('Hour_Group')['Calories_Burned']
mean_values_hour = grouped_hour.mean()
median_values_hour = grouped_hour.median()
plt.plot(mean_values_hour.index, median_values_hour.values, marker='o', label='Mean', color='#67AE6E')
plt.plot(median_values_hour.index, median_values_hour.values, marker='o', label='Median', color = '#FF9B17')
plt.title("Lượng calories sử dụng theo thời gian tập luyện", fontsize=16)
plt.xlabel("Thời gian tập luyện", fontsize=12)
plt.ylabel("Lượng calories sử dụng", fontsize=12)
plt.legend()
plt.tight_layout()
#plt.show()


# In[17]:


sns.displot(data_df["Workout_Type"],color= '#67AE6E', height=5, aspect=1, binwidth=1)
plt.title("Phân bố dữ liệu về hình thức tập luyện", fontsize=16)
plt.xlabel("Hình thức tập luyện", fontsize=12)
plt.ylabel("Số lượng", fontsize=12)
plt.tight_layout()
#plt.show()

# Tạo bảng đếm số lượng từng giới tính trong mỗi nhóm
Workout_type_counts = data_df.groupby(['Calories_Group', 'Workout_Type']).size().unstack()
# Tính phần trăm theo hàng (mỗi nhóm cộng lại = 100%)
Workout_type_percentage = Workout_type_counts.div(Workout_type_counts.sum(axis=1), axis=0) * 100
# Vẽ stacked bar chart
Workout_type_percentage.plot(kind='bar', stacked=True, color=['#67AE6E','#90C67C','#328E6E','#0D4715'])
plt.title("Lượng calories sử dụng theo hình thức tập luyện", fontsize=16)
plt.xlabel("Hình thức tập luyện", fontsize=12)
plt.ylabel("Lượng calories sử dụng", fontsize=12)
plt.tight_layout()
#plt.show()


# In[18]:


bins_frequency = [0.9, 1.9, 2.9, 3.9, 4.9, 5.9, 6.9, 10]
labels_frequency = ['1', '2', '3', '4', '5','6','7']
data_df['Frequency_Group'] = pd.cut(data_df['Workout_Frequency (days/week)'], bins=bins_frequency, labels=labels_frequency)
sns.displot(data_df['Frequency_Group'],color= '#67AE6E', height=5, aspect=1, binwidth=1)
plt.title("Phân bố dữ liệu về tần suất tập luyện", fontsize=16)
plt.xlabel("Tần suất tập luyện", fontsize=12)
plt.ylabel("Số lượng", fontsize=12)
plt.tight_layout()
#plt.show()

# Tạo bảng đếm số lượng từng giới tính trong mỗi nhóm
Frequency_counts = data_df.groupby(['Calories_Group', 'Workout_Frequency (days/week)']).size().unstack()
# Tính phần trăm theo hàng (mỗi nhóm cộng lại = 100%)
Frequency_percentage = Frequency_counts.div(Frequency_counts.sum(axis=1), axis=0) * 100
# Vẽ stacked bar chart
Frequency_percentage.plot(kind='bar', stacked=True, color=['#67AE6E','#90C67C','#328E6E','#0D4715'])
plt.title("Lượng calories sử dụng theo tần suất tập luyện", fontsize=16)
plt.xlabel("Tần suất tập luyện", fontsize=12)
plt.ylabel("Lượng calories sử dụng", fontsize=12)
plt.tight_layout()
#plt.show()

bins_level = [0.9, 1.9, 2.9, 5]
labels_level = ['1', '2', '3']
data_df['Level_Group'] = pd.cut(data_df['Experience_Level'], bins=bins_level, labels=labels_level)
sns.displot(data_df['Level_Group'],color= '#67AE6E', height=5, aspect=0.75, bins=3)
plt.title("Phân bố dữ liệu về mức độ thuần thục", fontsize=16)
plt.xlabel("Mức độ thuần thục", fontsize=12)
plt.ylabel("Số lượng", fontsize=12)
plt.tight_layout()
#plt.show()

# Tạo bảng đếm số lượng từng giới tính trong mỗi nhóm
Exp_level_counts = data_df.groupby(['Calories_Group', 'Experience_Level']).size().unstack()
# Tính phần trăm theo hàng (mỗi nhóm cộng lại = 100%)
Exp_level_percentage = Exp_level_counts.div(Exp_level_counts.sum(axis=1), axis=0) * 100
# Vẽ stacked bar chart
Exp_level_percentage.plot(kind='bar', stacked=True, color=['#67AE6E','#90C67C','#328E6E'])
plt.title("Lượng calories sử dụng theo mức độ thuần thuch", fontsize=16)
plt.xlabel("Mức độ thuần thục", fontsize=12)
plt.ylabel("Lượng calories sử dụng", fontsize=12)
plt.tight_layout()
#plt.show()


# In[19]:


###Chạy các mô hình ML (Encode trước rồi mới chia biến)
##Data encoding
data_df = data_df.drop(data_df.columns[10:], axis=1)
print (data_df [:5])


# In[20]:


print('Mã hóa lại các biến trong mô hình')
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

data_df['Gender'] = data_df['Gender'].astype(str)
data_df['Workout_Type'] = data_df['Workout_Type'].astype(str)

categorical_columns = ['Gender', 'Workout_Type']

preprocessor = ColumnTransformer(
    transformers=[
        ('', OneHotEncoder(), categorical_columns)
    ],
    remainder='passthrough'  # giữ lại các cột còn lại (Age)
)
data_df= preprocessor.fit_transform(data_df)

data_df = pd.DataFrame(
    data_df,
    columns = preprocessor.get_feature_names_out()
)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

print(data_df.head())


# In[21]:


corr_data_df = data_df.corr()
plt.figure(figsize=(15, 15))
sns.heatmap(corr_data_df, annot=True, cmap='Blues', fmt=".2f", linewidths=0.5) 
plt.title("Heatmap of Feature Correlation") 
plt.tight_layout()


# In[22]:


###Data splitting
print('Các biến trong mô hình')
print('Biến độc lập')
X = data_df.iloc[:,0:13].values
print(X[:5]) 

print('Biến phụ thuộc')
Y = data_df.iloc[:,13].values
print (Y [:5])


# In[23]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
print (X [:5])


# In[24]:


###Logistic regression


# In[25]:


from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# In[26]:


###Random tree forest
###Model training
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# Tập hợp các giá trị tham số cần thử
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20]
}

# Khởi tạo Random Forest
rf = RandomForestRegressor(random_state=42)

# GridSearchCV với K-Fold (cv=5)
grid_rf = GridSearchCV(rf, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_rf.fit(X_train, Y_train)

# In kết quả
print ('Kết quả từng tổ hợp:')
df = pd.DataFrame(grid_rf.cv_results_)
print(df[['params', 'mean_test_score', 'std_test_score']])
           
print("Tham số tốt nhất:", grid_rf.best_params_)
print("MSE trung bình tốt nhất:", -grid_rf.best_score_)

# Mô hình tốt nhất sau khi tìm được tham số tối ưu
best_model_rf = grid_rf.best_estimator_

# Dự đoán trên tập test
Y_pred_rf = best_model_rf.predict(X_test)

##Evaluation
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

###R2 score
r2_rf = r2_score(Y_test, Y_pred_rf)
print("R² score:", r2_rf)

####MAE
mae_rf = mean_absolute_error(Y_test, Y_pred_rf)
print(f'Mean Absolute Error (MAE): {mae_rf}')

####RMSE
rmse_rf = np.sqrt(mean_squared_error(Y_test, Y_pred_rf))
print(f'Root Mean Squared Error (RMSE): {rmse_rf}')

# Tính sai số
errors_rf = Y_test - Y_pred_rf

data_rf = pd.DataFrame({'Observed y': Y_test, 'Predicted y': Y_pred_rf})

# Vẽ biểu đồ Predicted vs Observed
plt.figure(figsize=(8, 6))
sns.regplot(x='Observed y', y='Predicted y', data=data_rf, scatter_kws={'s': 50, 'color': 'blue'}, line_kws={'color': 'gray'})
plt.title('Predicted vs Observed', fontsize=15)
plt.xlabel('Observed y', fontsize=12)
plt.ylabel('Predicted y', fontsize=12)
plt.show()


# In[27]:


###Gradient Boosting
###Model training
from sklearn.ensemble import GradientBoostingRegressor

# Tập hợp các giá trị tham số cần thử
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'learning_rate': [0.1, 0.3, 0.5]
}

# Khởi tạo Random Forest
gbr = GradientBoostingRegressor(random_state=42)

# GridSearchCV với K-Fold (cv=5)
grid_gbr = GridSearchCV(gbr, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_gbr.fit(X_train, Y_train)

# In kết quả
print("Kết quả từng tổ hợp:")
df = pd.DataFrame(grid_gbr.cv_results_)
print(df[['params', 'mean_test_score', 'std_test_score']])        
print("Tham số tốt nhất:", grid_gbr.best_params_)
print("MSE trung bình tốt nhất:", -grid_gbr.best_score_)

# Mô hình tốt nhất sau khi tìm được tham số tối ưu
best_model_gbr = grid_gbr.best_estimator_

# Dự đoán trên tập test
Y_pred_gbr = best_model_gbr.predict(X_test)

###R2 score
r2_gbr = r2_score(Y_test, Y_pred_gbr)
print("R² score:", r2_gbr)

####MAE
mae_gbr = mean_absolute_error(Y_test, Y_pred_gbr)
print(f'Mean Absolute Error (MAE): {mae_gbr}')

####RMSE
rmse_gbr =np.sqrt(mean_squared_error(Y_test, Y_pred_gbr))
print(f'Root Mean Squared Error (RMSE): {rmse_gbr}')

##Biểu đồ phân phối sai số
data_gbr = pd.DataFrame({'Observed y': Y_test, 'Predicted y': Y_pred_gbr})

# Vẽ biểu đồ Predicted vs Observed
plt.figure(figsize=(8, 6))
sns.regplot(x='Observed y', y='Predicted y', data=data_gbr, scatter_kws={'s': 50, 'color': 'blue'}, line_kws={'color': 'gray'})
plt.title('Predicted vs Observed', fontsize=15)
plt.xlabel('Observed y', fontsize=12)
plt.ylabel('Predicted y', fontsize=12)


# In[1]:


import joblib
import streamlit as st
import datetime
import os


# In[2]:


# File lưu trữ lịch sử
DATA_FILE = 'workout_history.csv'

# Hàm lưu dữ liệu mới vào file
def save_data(df):
    df.to_csv(DATA_FILE, index=False)


# In[3]:


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


# In[ ]:


if submitted:
    # Chuẩn bị dữ liệu cho mô hình
    X = [['Age', 'Gender', 'Weight (kg)', 'Avg_BPM', 'Session_Duration (hours)', 'Workout_Type', 'Workout_Frequency (days/week)',  'Experience_Level']]
    calories_pred = best_model_rf.predict(X)
    
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


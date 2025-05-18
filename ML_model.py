###Import dictionaries
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import joblib

###Import data set
data_df = pd.read_csv("C:/Users/admin/Downloads/gym_members_exercise_tracking.csv")
print('Mã hóa lại các biến trong mô hình')

data_df['Gender'] = data_df['Gender'].astype(str)
data_df['Workout_Type'] = data_df['Workout_Type'].astype(str)
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

X = data_df.iloc[:,0:13].values

print('Biến phụ thuộc')
Y = data_df.iloc[:,13].values

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

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
grid_rf = GridSearchCV(rf, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_rf.fit(X_train, Y_train)
best_model_rf = grid_rf.best_estimator_

pipeline = Pipeline(steps=[
    ('preprocessing', preprocessor), 
    ('chuẩn hóa',sc),
    ('regressor', best_model_rf.predict(X))
])

joblib.dump(pipeline, 'model.pkl')

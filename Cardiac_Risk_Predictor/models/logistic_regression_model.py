import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib as jl


# Load the dataset in a dataframe object
df = pd.read_csv('dataset\heart.csv')


# Data Preprocessing
#Check for duplicates
if (df.duplicated().any()):
    df.drop_duplicates()


# df.isnull().sum()

# Finding categorical values
categorical_values = []
numeric_values = []
for column in df.columns:
     if df[column].nunique() <=10:
          categorical_values.append(column)
     else:
          # df[column].fillna(0, inplace=True)
          numeric_values.append(column)


# Encoding categorical values
categorical_values.remove('sex')
categorical_values.remove('target')
df = pd.get_dummies(df, columns=categorical_values, drop_first=True)

# Feature Scaling
st = StandardScaler()
df[numeric_values] = st.fit_transform(df[numeric_values])


# Split dataset
dependent_value = 'target'
x = df.drop(dependent_value,axis=1)
y = df[dependent_value]


# Train test split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=42)


# Logistic Regression Classifier
log_reg = LogisticRegression()
log_reg.fit(x_train, y_train)


# Predict
y_pred = log_reg.predict(x_test)

# Check Accuracy
accuracy = accuracy_score(y_test,y_pred)
print(f"Accuracy Score for Logistic Regression Model :  {accuracy}")

# Save Model
jl.dump(log_reg, 'log_reg_model.pkl')
print("Model saved")

# Load Model
log_reg = jl.load('log_reg_model.pkl')

# Save the data columns from training set
model_columns = list(x.columns)
jl.dump(model_columns, 'log_reg_model_columns.pkl')
print("Model columns saved")
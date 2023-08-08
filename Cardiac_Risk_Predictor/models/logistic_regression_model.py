import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib as jl
import matplotlib.pyplot as plt 
import seaborn as sns
import pickle as pkl


# Load the dataset in a dataframe object
heart_df = pd.read_csv('dataset\heart.csv')

# Taking a look at first few rows
heart_df.head()

# Get basic information about the dataset
basicinfo = heart_df.info()

# Count of target values
heart_df['target'].value_counts()

# Get summary statistics
summary_stats = heart_df.describe()
print(summary_stats)

#Display data types of columns
column_data_types = heart_df.dtypes
print(column_data_types)

# Scatter plot for visualizing relationship of cholestrol and trestbps

plt.scatter(x = 'chol', y = 'trestbps', color = 'blue', data = heart_df)
plt.xlabel('cholestrol')                
plt.ylabel('trestbps')  
plt.title('Cholestrol Vs Trestbps') 


# Scatter plot for visualizing relationship of cholestrol and age

plt.scatter(x = 'chol', y = 'age', color = 'red', data = heart_df)
plt.xlabel('cholestrol')                
plt.ylabel('trestbps')  
plt.title('Cholestrol Vs Trestbps') 


# visualizing data distribution
# finding age wise distribution
plt.figure(figsize=(6, 4))
sns.displot(x="age", data= heart_df)
plt.title("Distribution of age")
plt.show()


# Data Preprocessing


#Check for duplicates
if (heart_df.duplicated().any()):
    heart_df.drop_duplicates()


heart_df.isnull().sum()


# Split dataset
dependent_value = 'target'
X = heart_df.drop(dependent_value,axis=1)
y = heart_df[dependent_value]


# Train test split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33,random_state=101)


# Standardize the features for better model performance
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)


# Logistic Regression Classifier
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)


# Predict
y_pred = log_reg.predict(X_test)

# Check Accuracy
accuracy = accuracy_score(y_test,y_pred)
print(f"Accuracy Score for Logistic Regression Model :  {accuracy}")

# Save Model
jl.dump(log_reg, 'log_reg_model.pkl')
print("Model saved")

# Save Scaler
with open('log_reg_scaler.pkl', 'wb') as scaler_file:
    pkl.dump(scaler, scaler_file)
print("Scaler saved")

# Save the data columns from training set
model_columns = list(X.columns)
jl.dump(model_columns, 'log_reg_model_columns.pkl')
print("Model columns saved")
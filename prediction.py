import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report

# Load HR data
df = pd.read_csv('HR-Employee-Attrition.csv')

# Title and introduction
st.title("Employee Attrition Prediction and Analysis")
st.write("This App allows you to predict employee attrition and explore HR data.")

# Sidebar for filtering options
st.sidebar.header("Filter Options")

# Filter by department
selected_department = st.sidebar.selectbox("Select Department", df["Department"].unique())
filtered_data = df[df["Department"] == selected_department]

# Show filtered data
st.sidebar.subheader("Filtered Data")
st.sidebar.write("Total Employees in", selected_department, "department:", len(filtered_data))

# Data Exploration Section
st.header("Data Exploration")

# Display the first few rows of the dataset
st.subheader("Sample Data")
st.write(filtered_data.head())

# Display dataset shape
st.subheader("Data Shape")
st.write("Number of Rows:", filtered_data.shape[0])
st.write("Number of Columns:", filtered_data.shape[1])

# Display data types and missing values
st.subheader("Data Information")
st.write(filtered_data.info())

# Display summary statistics
st.subheader("Summary Statistics")
st.write(filtered_data.describe())

# Boxplot for numeric features
st.subheader("Boxplot for Numeric Features")
numeric_cols = filtered_data.select_dtypes(include=[np.number]).columns
selected_feature = st.selectbox("Select a numeric feature for boxplot", numeric_cols)
if selected_feature:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Attrition', y=selected_feature, data=filtered_data)
    plt.xticks(rotation=90)
    st.pyplot()

# Bar chart for categorical features
st.subheader("Bar Chart for Categorical Features")
categorical_cols = filtered_data.select_dtypes(include=[np.object]).columns
selected_category = st.selectbox("Select a categorical feature for bar chart", categorical_cols)
if selected_category:
    plt.figure(figsize=(10, 6))
    sns.countplot(x=selected_category, hue='Attrition', data=filtered_data)
    plt.xticks(rotation=90)
    st.pyplot()

# Data Preprocessing Section
st.header("Data Preprocessing")

# Encode categorical variables
label_cols = filtered_data.select_dtypes(include=[np.object]).columns
label_mapping = {}
for col in label_cols:
    filtered_data[col], label_mapping[col] = pd.factorize(filtered_data[col])

# Model Building and Prediction
st.header("Employee Attrition Prediction")

if st.button("Predict Attrition"):
    X = filtered_data.drop(['Attrition'], axis=1)
    y = filtered_data['Attrition']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest model
    model = RandomForestClassifier(max_features='auto', n_estimators=100)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Display evaluation metrics
    st.subheader("Model Evaluation")
    st.write("Accuracy Score: ", accuracy_score(y_test, y_pred))
    st.write("ROC AUC Score: ", roc_auc_score(y_test, y_pred))
    st.write("Confusion Matrix: ", confusion_matrix(y_test, y_pred))
    st.write("Classification Report: ", classification_report(y_test, y_pred))

    # Display the most risky employees
    st.header("Most Risky Employees")
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    risky_employees = X_test.copy()
    risky_employees['Attrition Probability'] = y_pred_prob
    risky_employees = risky_employees.sort_values(by='Attrition Probability', ascending=False).head(10)
    st.dataframe(risky_employees)

# Data Visualization Section
st.header("Data Visualization")

# Show a pie chart of attrition rate
attrition_count = filtered_data['Attrition'].value_counts()
st.subheader("Attrition Rate")
st.write("Attrition Rate:")
st.write("No: ", attrition_count.get("No", 0))
st.write("Yes: ", attrition_count.get("Yes", 0))
plt.figure(figsize=(6, 6))
plt.pie(attrition_count, labels=['No', 'Yes'], autopct='%1.1f%%', startangle=140)
plt.axis('equal')
st.pyplot()


# Custom chart
st.subheader("Custom Chart")
data = filtered_data.groupby('JobRole')['MonthlyIncome'].mean()
st.bar_chart(data)

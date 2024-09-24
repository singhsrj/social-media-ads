import streamlit as st
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Load dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Train the model
classifier = DecisionTreeClassifier(criterion='log_loss', random_state=0)
classifier.fit(X_train, y_train)

# Streamlit UI

st.title("AdInsights AI ")
st.write("Harness AI to Know Your Audience and Predict Purchases with Every Ad.")

# Input fields
age = st.number_input("Enter Age", min_value=18, max_value=100, value=30)
salary = st.number_input("Enter Estimated Salary", min_value=10000, max_value=200000, value=60000)

# Prediction button
if st.button("Predict"):
    prediction = classifier.predict(sc.transform([[age, salary]]))
    result = "Will Buy" if prediction[0] == 1 else "Won't Buy"
    st.write(f"The model predicts: {result}")

# Display accuracy
y_pred = classifier.predict(X_test)
accuracy = np.mean(y_pred == y_test)
st.write(f"Model accuracy: {accuracy * 100:.2f}%")

import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
# from sklearn.metrics import accuracy_score, classification_report


st.header(
    "Maternal Health risk High risk , Mid risk and Low risk predictions")


health_data = pd.read_csv("Maternal Health Risk Data Set.csv")
describe = health_data.describe()
st.subheader("Data Description", divider="blue")
st.dataframe(describe)
st.subheader("Overview of the data", divider="blue")
st.dataframe(health_data)
fig, ax = plt.subplots()
ax.scatter(health_data.Age, health_data.HeartRate, color="red")
st.subheader("Distribution between the Age and the Heart rate", divider="blue")
st.pyplot(fig)
st.subheader("input data")
input_data = health_data.drop(columns="RiskLevel")
output_data = health_data["RiskLevel"]
st.dataframe(input_data)
input_train, input_test, output_train, output_test = train_test_split(
    input_data, output_data, test_size=0.3)
model_decisiontree = DecisionTreeClassifier()
model_decisiontree.fit(input_train, output_train)
predictions_decisionTree = model_decisiontree.predict(input_test)

algorithms = st.radio("Select an algorithms", [
    "Decision Tree", "Logistic Regression", "Owl"])

# Display content based on the selected option
if algorithms == "Decision Tree":
    st.header("Decision Tree")
    score_decisionTree = accuracy_score(output_test, predictions_decisionTree)
    st.write(score_decisionTree)


elif algorithms == "Logistic Regression":
    st.header("A dog")
    st.image("https://static.streamlit.io/examples/dog.jpg", width=200)
# elif algorithms == "Owl":
#     st.header("An owl")
#     st.image("https://static.streamlit.io/examples/owl.jpg", width=200)

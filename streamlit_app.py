import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
# from sklearn.metrics import accuracy_score, classification_report


st.header(
    "Maternal Health risk dataset: High risk , Mid risk and Low risk predictions")


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
model_logisticr = LogisticRegression()
model_KN = KNeighborsClassifier(n_neighbors=5)

model_decisiontree.fit(input_train, output_train)
model_logisticr.fit(input_train, output_train)
model_KN.fit(input_train, output_train)


predictions_decisionTree = model_decisiontree.predict(input_test)
predictions_logisticr = model_logisticr.predict(input_test)
predictions_KN = model_KN.predict(input_test)

algorithms = st.radio("Select an algorithms", [
    "Decision Tree", "Logistic Regression", "KN neighrest Neighbour", "Compare"])

# Display content based on the selected option
if algorithms == "Decision Tree":
    st.header("Decision Tree")

    score_decisionTree = accuracy_score(output_test, predictions_decisionTree)

# Round to two decimal places
    rounded_score_decisionTree = round(score_decisionTree, 2)

# Convert to a percentage
    percentage_score_decisionTree = rounded_score_decisionTree * 100

# Visualize the percentage in a Streamlit application
    st.metric(label="Accuracy Percentage", value=percentage_score_decisionTree)
    st.write("Accuracy Score Visualization")

# Define labels for the pie chart
    # Add more model names if needed
    labels = ["Decision Tree accuracy", "Not accurate"]

    # Define the percentage scores corresponding to the labels
    sizes = [percentage_score_decisionTree,
             100 - percentage_score_decisionTree]

    # Define colors for the sections of the pie chart
    colors = ['lightblue', 'lightgray']

    # Create the pie chart
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, colors=colors,
            autopct='%1.1f%%', startangle=90)
    # Equal aspect ratio ensures that pie is drawn as a circle.
    ax1.axis('equal')

    # Display the pie chart in Streamlit
    st.pyplot(plt)

elif algorithms == "Logistic Regression":
    st.header("Logic Regression")
    score_logisticr = accuracy_score(output_test, predictions_logisticr)
    rounded_score_logisticr = round(score_logisticr, 3)
    percentage_score_logisticr = rounded_score_logisticr * 100
    st.metric(label="Accuracy Percentage", value=percentage_score_logisticr)

    st.write("Accuracy Score Visualization")

   # Define labels for the pie chart
    # Add more model names if needed
    labels = ["Decision Tree accuracy", "Not accurate"]

    # Define the percentage scores corresponding to the labels
    sizes = [percentage_score_logisticr,
             100 - percentage_score_logisticr]

    # Define colors for the sections of the pie chart
    colors = ['lightgreen', 'lightgray']

    # Create the pie chart
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, colors=colors,
            autopct='%1.1f%%', startangle=90)
    # Equal aspect ratio ensures that pie is drawn as a circle.
    ax1.axis('equal')

    # Display the pie chart in Streamlit
    st.pyplot(plt)
elif algorithms == "KN neighrest Neighbour":
    st.header("KN neighrest Neighbour")

    score_KN = accuracy_score(output_test, predictions_KN)
    rounded_score_KN = round(score_KN, 3)
    percentage_score_KN = rounded_score_KN * 100
    st.metric(label="Accuracy Percentage", value=percentage_score_KN)

    st.write("Accuracy Score Visualization")

   # Define labels for the pie chart
    # Add more model names if needed
    labels = ["Decision Tree accuracy", "Not accurate"]

    # Define the percentage scores corresponding to the labels
    sizes = [percentage_score_KN,
             100 - percentage_score_KN]

    # Define colors for the sections of the pie chart
    colors = ['lightyellow', 'lightgray']

    # Create the pie chart
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, colors=colors,
            autopct='%1.1f%%', startangle=90)
    # Equal aspect ratio ensures that pie is drawn as a circle.
    ax1.axis('equal')

    # Display the pie chart in Streamlit
    st.pyplot(plt)


def compare_accuracy_scores():
    # Calculate the accuracy scores for different models
    score_logisticr = accuracy_score(output_test, predictions_logisticr)
    score_KN = accuracy_score(output_test, predictions_KN)
    score_decisionTree = accuracy_score(output_test, predictions_decisionTree)
    accuracy_scores = [score_decisionTree, score_logisticr, score_KN]
    models = ['Decision Tree', 'Logistic Regression', 'kN Nearest Neighbour']

    # Create a bar chart
    plt.bar(models, accuracy_scores, color=['blue', 'green', 'red'])
    plt.xlabel('Model')
    plt.ylabel('Accuracy Score')
    plt.title('Comparison of Accuracy Scores for Different Models')
    plt.ylim(0, 1)  # Set the y-axis range to be between 0 and 1 for accuracy.

    # Display the accuracy scores on top of the bars.
    for i, score in enumerate(accuracy_scores):
        plt.text(i, score, f'{score:.2f}', ha='center', va='bottom')

    # Show the bar chart in the Streamlit app
    st.pyplot(plt)


# Streamlit app code
if algorithms == "Compare":
    st.header("Comparison of Accuracy Scores for Different Models")
    compare_accuracy_scores()

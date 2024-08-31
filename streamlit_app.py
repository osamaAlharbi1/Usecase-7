import streamlit as st
import requests

# Set the URL for the FastAPI app
url_kmeans = "https://usecase-7-12ia.onrender.com/predict_kmeans"  # Update with your actual endpoint
url_dbscan = "https://usecase-7-12ia.onrender.com/predict_dbscan"  # Update with your actual endpoint

# Create the Streamlit app
st.title("Player Value and Performance Clustering")

# Input fields for the user to provide data
current_value = st.number_input("Current Value (in thousands)", min_value=0.0, max_value=100000.0, value=50000.0)
goals = st.number_input("Goals", min_value=0, max_value=100, value=10)
age = st.number_input("Age", min_value=18, max_value=40, value=25)
award = st.number_input("Awards", min_value=0, max_value=10, value=1)

# Prepare the data to be sent as JSON
data = {
    "current_value": current_value,
    "goals": goals,
    "age": age,
    "award": award
}

# Button to trigger the KMeans prediction
if st.button("Predict KMeans Cluster"):
    # Send the data to FastAPI and get the response
    response = requests.post(url_kmeans, json=data)
    if response.status_code == 200:
        result = response.json()
        st.success(f"The predicted KMeans cluster is: {result['kmeans_pred']}")
    else:
        st.error("Failed to get a KMeans prediction. Please try again.")

# Button to trigger the DBSCAN prediction
if st.button("Predict DBSCAN Cluster"):
    # Send the data to FastAPI and get the response
    response = requests.post(url_dbscan, json=data)
    if response.status_code == 200:
        result = response.json()
        st.success(f"The predicted DBSCAN cluster is: {result['dbscan_pred']}")
    else:
        st.error("Failed to get a DBSCAN prediction. Please try again.")

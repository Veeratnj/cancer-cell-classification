import streamlit as st
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the pre-trained model
model = joblib.load('random_forest_cancer_cell_classification.joblib')

# Load the scaler (if you saved the scaler)
scaler = joblib.load('scaler.pkl')

# Function to predict based on user input
def predict(input_data):
    # Preprocess input data (scale it)
    input_data_scaled = scaler.transform([input_data])
    
    # Make prediction
    prediction = model.predict(input_data_scaled)
    probability = model.predict_proba(input_data_scaled)
    
    # Return prediction and probability
    return prediction[0], probability[0]

# Streamlit UI setup
st.title("Cancer Cell Classification")

# Create 30 input boxes for the features
input_data = []
for i in range(1, 31):
    input_data.append(st.number_input(f"Feature {i}", min_value=0.0, step=0.1))

# When the user clicks the predict button
if st.button("Predict"):
    # Convert input data to a numpy array
    input_data = np.array(input_data)
    
    # Get prediction and probability
    prediction, probability = predict(input_data)
    
    # Show the prediction
    if prediction == 0:
        st.write("The cell is predicted to be Benign (Healthy).")
    else:
        st.write("The cell is predicted to be Malignant (Cancerous).")
    
    # Show the probability
    st.write(f"Probability of Benign: {probability[0]:.2f}")
    st.write(f"Probability of Malignant: {probability[1]:.2f}")

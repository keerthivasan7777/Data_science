import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import streamlit as st



# Load dataset and encode categorical variables

@st.cache
def load_data():
    dataset = pd.read_csv('C:\\Users\\Keerthivasan R\\OneDrive\\Desktop\\Internship_DS_Project\\cleaned_data.csv', encoding='latin1')
    
    # Encode categorical columns
    encoders = {}
    for column in ['Position', 'Location', 'Gender', 'Education']:
        encoder = LabelEncoder()
        dataset[column] = encoder.fit_transform(dataset[column])
        encoders[column] = encoder
        
    # Create Position_Location feature
    dataset['Position_Location'] = dataset['Position'] * dataset['Location']
    
    return dataset, encoders



# Load the dataset and encode categorical columns
dataset, encoders = load_data()

# Decode Position and Location for user input (to make it user-friendly)
position_decoder = encoders['Position']
location_decoder = encoders['Location']

# Feature and target selection
x = dataset[['Position', 'Location', 'Position_Location']]
y = dataset.iloc[:, -1].values  # Assuming the last column is the salary

# Normalize the target variable
scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y_scaled, test_size=0.30,
                                                     random_state=0)

# Scale the features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Train Linear Regression model
regressor = LinearRegression()
regressor.fit(x_train_scaled, y_train)

# Evaluate the model
train_score = regressor.score(x_train_scaled, y_train)
test_score = regressor.score(x_test_scaled, y_test)

# Display scores
st.write(f"Training Score: {train_score}")
st.write(f"Test Score: {test_score}")

# User input for prediction
st.title('Salary Prediction')
st.write('Please enter the following details for prediction:')

# Select Position and Location from dropdowns (decoded values for the user)
position_input = st.selectbox("Select Position:", position_decoder.classes_)
location_input = st.selectbox("Select Location:", location_decoder.classes_)

# Encode the selected values before prediction
encoded_position = position_decoder.transform([position_input])[0]
encoded_location = location_decoder.transform([location_input])[0]

# Create Position_Location feature
position_location_input = encoded_position * encoded_location

# Prepare the input for prediction
user_input = np.array([[encoded_position, encoded_location, position_location_input]])
user_input_scaled = scaler.transform(user_input)

# Predict salary
if st.button("Predict Salary"):
    prediction_scaled = regressor.predict(user_input_scaled)
    prediction = scaler_y.inverse_transform(prediction_scaled.reshape(-1, 1)).flatten()

    # Convert salary to Lakhs
    salary_in_lakhs = prediction[0]
    
    st.write(f"Predicted Salary: ₹{salary_in_lakhs:.2f} Lakhs")




#SENT THIS PROJECT TO MY MAIL

#keerthivasan
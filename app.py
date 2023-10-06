import streamlit as st
import pandas as pd
import pickle

# Load the saved Random Forest model
rf_model = pickle.load(open('knn_model_crime_name2.pkl', 'rb'))

# Load the saved OneHotEncoder
enc = pickle.load(open('one_hot_encoder.pkl', 'rb'))

# Set the background color to dark blue

st.title('Crime Prediction App')

st.subheader('Enter the following details to predict crime')

# Get input from the user
date_time = st.text_input('Date and Time (2023-05-14 12:30:00)')
latitude = st.text_input('Latitude')
longitude = st.text_input('Longitude')
city = st.text_input('City')
state = st.text_input('State')

# Create a button for prediction
if st.button('Predict'):
    # Check if input is not empty
    if not all([date_time, latitude, longitude, city, state]):
        st.error('Please enter all the details to predict crime')
    else:
        # Create a sample input from the user
        sample_input = pd.DataFrame({
            'date_time': [date_time],
            'latitude': [float(latitude)],
            'longitude': [float(longitude)],
            'city': [city],
            'state': [state]
        })

        # Preprocess the sample input
        sample_input['date_time'] = pd.to_datetime(sample_input['date_time'])
        sample_input['year'] = sample_input['date_time'].dt.year
        sample_input['month'] = sample_input['date_time'].dt.month
        sample_input['day'] = sample_input['date_time'].dt.day

        # Perform one-hot encoding for categorical variables
        cat_cols = ['city', 'state']
        encoded_cols = pd.DataFrame(enc.transform(sample_input[cat_cols]))
        encoded_cols.columns = enc.get_feature_names(cat_cols)

        # Handle unknown categories dynamically
        missing_categories = set(sample_input[cat_cols].values.flatten()) - set(enc.categories_[0])
        if missing_categories:
            for category in missing_categories:
                encoded_cols[category] = 0

        sample_input.drop(cat_cols, axis=1, inplace=True)
        sample_input = pd.concat([sample_input, encoded_cols], axis=1)

        # Make predictions using the Random Forest model
        rf_prediction = rf_model.predict(sample_input)

        # Display the result
        st.subheader('Crime Prediction Result')
        if rf_prediction[0] == 0:
            st.write('No crime is predicted.')
        else:
            st.write('A crime is predicted:', rf_prediction[0])

import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the XGBoost model
xgboost_model = joblib.load('assignment_model_2_final.pkl')

# Function to preprocess the input data
def preprocess_data(data):
    # Features to be scaled
    features_to_scale = ['Major_Axis_Length', 'Minor_Axis_Length', 'Eccentricity', 'Solidity', 'Extent', 'Roundness', 'Aspect_Ratio', 'Compactness']

    # Apply standard scaling to the selected features
    scaler = StandardScaler()
    data[features_to_scale] = scaler.fit_transform(data[features_to_scale])

    return data

# Function to make predictions
def predict(data):
    # Preprocess the input data
    data_processed = preprocess_data(data)

    # Make predictions
    predictions = xgboost_model.predict(data_processed)

    return predictions

# Streamlit UI
st.header('Pumpkin Seed Quality Classification App')
st.write("Pumpkin seeds are a popular snack in Turkey and deeply rooted in Turkey's cuisine. But pumpkin seeds do not equal pumpkin seeds. The two most important quality types of pumpkin seeds in Turkey are “Ürgüp Sivrisi” and “Çerçevelik”, generally grown in the regions Ürgüp and Karacaören in Turkey.")
st.write('Enter the pumpkin seed measurements below to find out of which quality type your probe most likely is:')


average_major_axis_length = 456.60
average_minor_axis_length = 225.79
average_eccentricity = 0.86
average_solidity = 0.99
average_extent = 0.69
average_roundness = 0.79
average_aspect_ratio = 2.04
average_compactness = 0.70
# Input form with input fields

major_axis_length = st.number_input('Major Axis Length', min_value=0.0, value=0.0)
st.markdown('<p style="font-size: smaller; font-style: italic;">The average Major Axis Length is {:.2f}</p>'.format(average_major_axis_length), unsafe_allow_html=True)

minor_axis_length = st.number_input('Minor Axis Length', min_value=0.0, value=0.0)
st.markdown('<p style="font-size: smaller; font-style: italic;">The average Minor Axis Length is {:.2f}</p>'.format(average_minor_axis_length), unsafe_allow_html=True)

eccentricity = st.number_input('Eccentricity', min_value=0.0, max_value=1.0, value=0.0)
st.markdown('<p style="font-size: smaller; font-style: italic;">The average Eccentricity is {:.2f}. You can maximally input a value of 1.00.</p>'.format(average_eccentricity), unsafe_allow_html=True)

solidity = st.number_input('Solidity', min_value=0.0, max_value=1.0, value=0.0)
st.markdown('<p style="font-size: smaller; font-style: italic;">The average Solidity is {:.2f}. You can maximally input a value of 1.00.</p>'.format(average_solidity), unsafe_allow_html=True)

extent = st.number_input('Extent', min_value=0.0, max_value=1.0, value=0.0)
st.markdown('<p style="font-size: smaller; font-style: italic;">The average Extent is {:.2f}. You can maximally input a value of 1.00.</p>'.format(average_extent), unsafe_allow_html=True)

roundness = st.number_input('Roundness', min_value=0.0, max_value=1.0, value=0.0)
st.markdown('<p style="font-size: smaller; font-style: italic;">The average Roundness is {:.2f}. You can maximally input a value of 1.00.</p>'.format(average_roundness), unsafe_allow_html=True)

aspect_ratio = st.number_input('Aspect Ratio', min_value=0.0, value=0.0)
st.markdown('<p style="font-size: smaller; font-style: italic;">The average Aspect Ratio is {:.2f}</p>'.format(average_aspect_ratio), unsafe_allow_html=True)

compactness = st.number_input('Compactness', min_value=0.0, max_value=1.0, value=0.0)
st.markdown('<p style="font-size: smaller; font-style: italic;">The average Compactness is {:.2f}. You can maximally input a value of 1.00.</p>'.format(average_compactness), unsafe_allow_html=True)


if not major_axis_length or not minor_axis_length or not eccentricity or not solidity or not extent or not roundness or not aspect_ratio or not compactness:
    st.error('Error: All fields are mandatory. Please fill in all measurements.')
else:
    user_input = pd.DataFrame({
        'Major_Axis_Length': [major_axis_length],
        'Minor_Axis_Length': [minor_axis_length],
        'Eccentricity': [eccentricity],
        'Solidity': [solidity],
        'Extent': [extent],
        'Roundness': [roundness],
        'Aspect_Ratio': [aspect_ratio],
        'Compactness': [compactness]
    })



    # Make predictions on user input
    if st.button('Predict'):
        result = predict(user_input)

        # Map the prediction to the corresponding class
        class_mapping = {0: 'Çerçevelik', 1: 'Ürgüp Sivrisi'}
        prediction_label = class_mapping[result[0]]

        st.write('This pumpkin seed is of quality type', prediction_label+'.')


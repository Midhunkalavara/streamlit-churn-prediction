import numpy as np
import joblib
import streamlit as st
from sklearn.preprocessing import PowerTransformer

# Load the saved churn prediction model
loaded_model = joblib.load(open('xgboost_model.pkl', 'rb'))

# Function to transform data
def transform_data(input_data):
    # Initialize PowerTransformers
    pt_yeo = PowerTransformer(method='yeo-johnson')
    pt_boxcox = PowerTransformer(method='box-cox')

    # Transform specific variables using Yeo-Johnson
    yeo_johnson_values = np.array([input_data[1], input_data[2], input_data[3]]).reshape(-1, 1)
    yeo_johnson_transformed = pt_yeo.fit_transform(yeo_johnson_values).flatten()

    # Transform other numerical variables using Box-Cox
    other_vars = np.array([input_data[i] for i in range(len(input_data)) if i not in [1, 2, 3]]).reshape(-1, 1)
    boxcox_transformed = pt_boxcox.fit_transform(0.001 + other_vars).flatten()

    # Combine transformed data into a single array
    transformed_data = np.concatenate([yeo_johnson_transformed, boxcox_transformed])
    return transformed_data

# Function for prediction
def churn_prediction(input_data):
    # Transform input data
    transformed_data = transform_data(input_data)

    # Reshape the array for a single instance
    input_data_reshaped = transformed_data.reshape(1, -1)

    # Make a prediction using the loaded model
    prediction = loaded_model.predict(input_data_reshaped)

    # Return the prediction result
    if prediction[0] == 0:
        return 'The customer will not churn.'
    else:
        return 'The customer will churn.'

# Main function to create the web app
def main():
    # Web app title
    st.title('Customer Churn Prediction Web App')

    # Collect input data from the user
    purchased_approved = st.text_input('Purchased Approved')
    delivered_estimated = st.text_input('Delivered Estimated')
    purchased_delivered = st.text_input('Purchased Delivered')
    price = st.text_input('Price')
    freight_value = st.text_input('Freight Value')
    product_weight_g = st.text_input('Product Weight (g)')
    payment_installments = st.text_input('Payment Installments')
    monetary = st.text_input('Monetary Value')
    payment_type_credit_card = st.text_input('Payment Type (Credit Card)')
    payment_type_debit_card = st.text_input('Payment Type (Debit Card)')

    # Diagnosis variable
    diagnosis = ''

    # Button to trigger prediction
    if st.button('Churn Prediction Result'):
        try:
            # Convert inputs to float for processing
            input_data = list(map(float, [
                purchased_approved,
                delivered_estimated,
                purchased_delivered,
                price,
                freight_value,
                product_weight_g,
                payment_installments,
                monetary,
                payment_type_credit_card,
                payment_type_debit_card
            ]))

            diagnosis = churn_prediction(input_data)
        except ValueError:
            diagnosis = 'Please enter valid numerical inputs for all fields.'

    # Display the result
    st.success(diagnosis)

if __name__ == '__main__':
    main()

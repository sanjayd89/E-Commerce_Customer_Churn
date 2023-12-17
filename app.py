import numpy as np
import pandas as pd
import streamlit as st 
from joblib import load


model_path = 'customer_churn_best_model.joblib'

st.title('Customer Churn Prediction')

def get_prediction(data, model_path):

    #load data
    infer_df = data

    #Processing before Prediction
    infer_df['PreferredLoginDevice'] = infer_df['PreferredLoginDevice'].replace('Phone', 'Mobile').replace('Mobile Phone', 'Mobile')
    infer_df['PreferredPaymentMode'] = infer_df['PreferredPaymentMode'].replace('Cash on Delivery', 'COD').replace('CC', 'Credit Card')
    infer_df['PreferedOrderCat'] = infer_df['PreferedOrderCat'].replace('Mobile Phone', 'Mobile')

    #load the model
    estimator = load(model_path)

    #predict based on the loaded model
    prediction = estimator.predict(infer_df)[0]

    #Provide Result
    result = 'Customer will CHURN' if prediction == 1 else 'Customer will STAY'    
    return prediction


def main():
    
    Tenure = st.selectbox("Enter Tenure", np.arange(0.0, 50.0))
    PreferredLoginDevice = st.radio("Preferred login device of customer", ("Mobile", "Computer"))
    CityTier = st.radio("City Tier", (1, 2, 3))
    WarehouseToHome = st.selectbox("Distance in between warehouse to home of customer", np.arange(0.0, 200.0))
    PreferredPaymentMode = st.selectbox('Preferred payment method of customer', 
    ['Debit Card', 'Credit Card', 'UPI', 'E wallet', 'Cash on Delivery']) 
    Gender = st.radio("Customer Gender", ("Male", "Female"))
    HourSpendOnApp = st.selectbox("Number of hours spend on mobile application or website", np.arange(0.0, 10.0))
    NumberOfDeviceRegistered = st.selectbox("Total number of deceives is registered on particular customer", np.arange(1, 11))
    PreferedOrderCat = st.selectbox('Preferred order category of customer in last month', 
    ['Laptop & Accessory', 'Mobile', 'Others', 'Fashion', 'Grocery'])
    SatisfactionScore = st.slider("Satisfactory score of customer on service", 1, 5, 1)
    MaritalStatus = st.radio("Marital status of customer", ("Married", "Single", "Divorced"))
    NumberOfAddress = st.selectbox("Total number of address added on particular customer", np.arange(1, 50))
    Complain = st.radio("Any complaint has been raised in last month", ("No", "Yes"))
    Complain = 1 if Complain == "Yes" else 0
    OrderAmountHikeFromlastYear = st.selectbox("Percentage increases in order from last year", np.arange(0.0, 100.0))
    CouponUsed = float(st.selectbox("Total number of coupon has been used in last month", np.arange(0, 50.0)))
    OrderCount = st.selectbox("Total number of orders has been places in last month", np.arange(0, 50))
    DaySinceLastOrder = st.selectbox("Day Since last order by customer", np.arange(0, 100))
    CashbackAmount = st.selectbox("Average cashback in last month", np.arange(0.0, 400.0))

    data_dict = {
        'Tenure' : [Tenure], 'PreferredLoginDevice' : [PreferredLoginDevice],'CityTier' : [CityTier],'WarehouseToHome' : [WarehouseToHome],'PreferredPaymentMode' : [PreferredPaymentMode],'Gender' : [Gender],'HourSpendOnApp' : [HourSpendOnApp],'NumberOfDeviceRegistered' : [NumberOfDeviceRegistered],'PreferedOrderCat' : [PreferedOrderCat],'SatisfactionScore' : [SatisfactionScore],'MaritalStatus' : [MaritalStatus],'NumberOfAddress' : [NumberOfAddress],'Complain' : [Complain],'OrderAmountHikeFromlastYear' : [OrderAmountHikeFromlastYear],'CouponUsed' : [CouponUsed],'OrderCount' : [OrderCount],'DaySinceLastOrder' : [DaySinceLastOrder],'CashbackAmount' : [CashbackAmount]
    }

    user_data = pd.DataFrame(data_dict)

    churn_prediction = ""
    if st.button("Predict"):
        churn_prediction = get_prediction(user_data, model_path)

    if churn_prediction == 0:
        st.success("The Customer will STAY")
    elif churn_prediction == 1:
        st.error("The Customer will LEAVE")


if __name__=='__main__':
    main()

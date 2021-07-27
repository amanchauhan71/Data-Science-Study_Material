from flask import Flask, request
import pandas as pd
import numpy as np
import pickle
from flasgger import Swagger
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
Swagger(app)
pickle_in = open('blackfriday.pkl', 'rb')
Regression = pickle.load(pickle_in)


@app.route("/")
def welcome():
    return "Welcome All"


@app.route("/predict")
def predict_student_performance():
    """Let's Authenticate the Student Performance
         This is using docstrings for specifications.
         ---
         parameters:
           - name: Age
             in: query
             type: string
             required: true
           - name: Occupation
             in: query
             type: string
             required: true
           - name: Stay_In_Current_City_Years
             in: query
             type: string
             required: true
           - name: Marital_Status
             in: query
             type: string
             required: true
           - name: Product_Category_1
             in: query
             type: number
             required: true
           - name: Product_Category_2
             in: query
             type: string
             required: true
           - name: Product_Category_3
             in: query
             type: string
             required: true
           - name: Gender
             in: query
             type: string
             required: true
           - name: City_Category
             in: query
             type: string
             required: true
         responses:
             200:
                 description: The output values

         """
    Age = int(request.args.get('Age'))
    Occupation = int(request.args.get('Occupation'))
    Stay_In_Current_City_Years = request.args.get('Stay_In_Current_City_Years')
    Marital_Status = request.args.get('Marital_Status')
    Product_Category_1 = request.args.get('Product_Category_1')
    Product_Category_2 = request.args.get('Product_Category_2')
    Product_Category_3 = request.args.get('Product_Category_3')

    Gender = request.args.get('Gender')

    if Gender == "Male":
        Gender_M = 1
    else:
        Gender_M = 0

    City_Category = request.args.get('City_Category')

    if City_Category == "A":
        City_Category_B = 0
        City_Category_C = 0
    elif City_Category == "B":
        City_Category_B = 1
        City_Category_C = 0
    else:
        City_Category_B = 0
        City_Category_C = 1

    print(Occupation)

    print(Age)

    filename_scaler = 'scaler_model.pickle'
    scaler_model = pickle.load(open(filename_scaler, 'rb'))
    scaled_data = scaler_model.transform([
        [Age, Occupation, Stay_In_Current_City_Years, Marital_Status, Product_Category_1, Product_Category_2,
         Product_Category_3, Gender_M,
         City_Category_B, City_Category_C]])

    # print(Age, Occupation, Stay_In_Current_City_Years, Marital_Status, Product_Category_1, Product_Category_2,
    #       Product_Category_3, Gender_M,
    #       City_Category_B, City_Category_C)

    prediction = Regression.predict(scaled_data)
    return "The prediction value is " + str(prediction)


if __name__ == '__main__':
    app.run(debug=True)

# import pickle
# import numpy as np
# import pandas as pd
# from flask import Flask, render_template, request
#
# app = Flask(__name__)
# model = pickle.load(open("LinearRegressionModel.pkl", 'rb'))
# car = pd.read_csv('Cleaned_Car.csv')
#
#
# @app.route('/')
# def index():
#     companies = sorted(car['company'].unique())
#     car_models = car[['company', 'name']].to_dict(orient='records')
#     years = sorted(car['year'].unique(), reverse=True)
#     fuel_types = car['fuel_type'].unique()
#     return render_template('index.html', companies=companies, car_models=car_models, years=years, fuel_types=fuel_types)
#
# @app.route('/predict', methods=['POST'])
# def predict():
#     company = request.form.get('company')
#     car_model = request.form.get('car_model')
#     year = int(request.form.get('year'))
#     fuel_type = request.form.get('fuel_type')
#     kms_driven = int(request.form.get('kilo_driven'))
#
#     prediction = model.predict(pd.DataFrame([[car_model, company, year, kms_driven, fuel_type]], columns=['name', 'company', 'year', 'kms_driven', 'fuel_type']))
#     return str(round(prediction[0], 2))
#
# if __name__ == "__main__":
#     app.run(debug=True)


from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)
car = pd.read_csv('Cleaned_Car.csv')
model = joblib.load('LinearRegressionModel.pkl')


@app.route('/')
def index():
    companies = sorted(car['company'].unique())
    car_models = car[['company', 'name']].to_dict(orient='records')
    years = sorted(car['year'].unique(), reverse=True)
    fuel_types = car['fuel_type'].unique()
    return render_template('index.html', companies=companies, car_models=car_models, years=years, fuel_types=fuel_types)


@app.route('/predict', methods=['POST'])
def predict():
    company = request.form.get('company')
    car_model = request.form.get('car_model')
    year = int(request.form.get('year'))
    kms_driven = int(request.form.get('kilo_driven'))
    fuel_type = request.form.get('fuel_type')

    input_data = pd.DataFrame([[car_model, company, year, kms_driven, fuel_type]],
                              columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'])
    prediction = model.predict(input_data)[0]

    return str(prediction)


if __name__ == '__main__':
    app.run(debug=True)

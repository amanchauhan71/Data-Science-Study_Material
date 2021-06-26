from flask import Flask, request
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
pickle_in = open('Classifier.pkl', 'rb')
classifier = pickle.load(pickle_in)

test_dict = {'GOOBU': 158,
             'QOQTS': 149,
             'UKPGS': 128,
             'DNQDD': 122,
             'GJJHK': 118,
             'ZOWMK': 117,
             'VVTVA': 114,
             'KZKKE': 111,
             'CCAAW': 109,
             'CUQAM': 107,
             'VKWQH': 100,
             'IDGFP': 94,
             'UAGPU': 87,
             'UUUQX': 84,
             'OJOBU': 81,
             'CIMBB': 74,
             'ZMNYA': 69,
             'GOKXL': 64,
             'LAYPA': 57,
             'KFZMY': 52,
             'VHDHF': 51,
             'FBUMG': 46,
             'ANKYI': 41}


@app.route("/")
def welcome():
    return "Welcome All"


@app.route("/predict")
def predict_student_performance():
    school = request.args.get('school')

    for key in test_dict:
        if key == school:
            school = int(test_dict[key])
            break

    school_setting = request.args.get('school_setting')

    if school_setting == "Urban":
        school_setting_Suburban = 0
        school_setting_Urban = 1
    elif school_setting == "Suburban":
        school_setting_Suburban = 1
        school_setting_Urban = 0
    else:
        school_setting_Suburban = 0
        school_setting_Urban = 0

    school_type = request.args.get('school_type')

    if school_type == 'Public':
        school_type_Public = 1
    else:
        school_type_Public = 0

    teaching_method = request.args.get('teaching_method')

    if teaching_method == "Standard":
        teaching_method_Standard = 1
    else:
        teaching_method_Standard = 0

    n_student = request.args.get('n_student')
    gender = request.args.get('gender')

    if gender == "Male":
        gender_Male = 1
    else:
        gender_Male = 0

    lunch = request.args.get('lunch')

    if lunch == 'qualify':
        lunch_Qualifies = 1
    else:
        lunch_Qualifies = 0
    pretest = float(request.args.get('pretest'))

    scaler = StandardScaler()
    filename_scaler = 'scaler_model.pickle'
    scaler_model = pickle.load(open(filename_scaler, 'rb'))
    scaled_data = scaler_model.transform([
        [n_student, pretest, school, school_setting_Suburban, school_setting_Urban, school_type_Public,
         teaching_method_Standard,
         gender_Male, lunch_Qualifies]])

    # print(n_student,pretest,school, school_setting_Suburban, school_setting_Urban, school_type_Public, teaching_method_Standard,
    #        gender_Male, lunch_Qualifies)
    #
    # print(scaled_data)
    prediction = classifier.predict(scaled_data)
    return "The prediction value is " + str(prediction)


if __name__ == '__main__':
    app.run(debug=True)

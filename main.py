from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

# ------------------ LOAD DATA ------------------
training_data = pd.read_csv("datasets/Training.csv")
precautions = pd.read_csv("datasets/precautions_df.csv")
workout = pd.read_csv("datasets/workout_df.csv")
description = pd.read_csv("datasets/description.csv")
medications = pd.read_csv("datasets/medications.csv")
diets = pd.read_csv("datasets/diets.csv")

# ------------------ LOAD MODEL ------------------
svc = pickle.load(open('Model/svc.pkl', 'rb'))

# ------------------ FEATURE CREATION ------------------
X = training_data.drop('prognosis', axis=1)
symptoms_list = X.columns.values.tolist()
symptoms_dict = {symptom: index for index, symptom in enumerate(symptoms_list)}

# ------------------ DISEASE MAPPING ------------------
disease_list = training_data['prognosis'].unique()

# ------------------ HELPER ------------------
def helper(dis):
    dis = str(dis).strip().lower()

    desc = description[description['Disease'].str.lower() == dis]['Description']
    desc = " ".join(desc) if not desc.empty else "No description available"

    pre_df = precautions[precautions['Disease'].str.lower() == dis]
    pre = pre_df[['Precaution_1','Precaution_2','Precaution_3','Precaution_4']].values.tolist()[0] if not pre_df.empty else ["No precautions"]

    med = medications[medications['Disease'].str.lower() == dis]['Medication'].tolist() or ["No medication"]

    die = diets[diets['Disease'].str.lower() == dis]['Diet'].tolist() or ["No diet"]

    wrk = workout[workout['disease'].str.lower() == dis]['workout'].tolist() or ["No workout"]

    return desc, pre, med, die, wrk


# ------------------ PREDICT ------------------
def get_predicted_value(symptoms):
    input_vector = np.zeros(len(symptoms_dict))

    for s in symptoms:
        if s in symptoms_dict:
            input_vector[symptoms_dict[s]] = 1

    pred = svc.predict([input_vector])[0]
    return disease_list[pred]


# ------------------ ROUTES ------------------

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    symptoms = request.form.get('symptoms')

    if not symptoms:
        return render_template('index.html', message="Please enter symptoms")

    user_symptoms = [s.strip() for s in symptoms.split(',')]

    predicted_disease = get_predicted_value(user_symptoms)

    dis_des, precautions_list, meds, rec_diet, wrkout = helper(predicted_disease)

    return render_template(
        'index.html',
        predicted_disease=predicted_disease,
        dis_des=dis_des,
        my_precautions=precautions_list,
        medications=meds,
        my_diet=rec_diet,
        workout=wrkout
    )


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/contact')
def contact():
    return render_template('contact.html')


@app.route('/developer')
def developer():
    return render_template('developer.html')


@app.route('/blog')
def blog():
    return render_template('blog.html')


if __name__ == '__main__':
    app.run(debug=True)
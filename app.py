from flask import Flask, request, render_template_string
import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

DATASET = "ObesityDataSet_raw_and_data_sinthetic.csv"
df = pd.read_csv(DATASET)

print("=" * 60)
print("EDA")
print("=" * 60)

print("Shape:", df.shape)
print("\nColumns")
print(df.columns.tolist())

print("\nInfo")
print(df.info())

print("\nMissing Values")
print(df.isnull().sum())

print("\nDuplicates:", df.duplicated().sum())

df = df.drop_duplicates()

print("\nStatistical Summary")
print(df.describe(include="all"))

target = "NObeyesdad"

encoders = {}

categorical_columns = df.select_dtypes(include="object").columns.tolist()

for col in categorical_columns:
    encoder = LabelEncoder()
    df[col] = encoder.fit_transform(df[col])
    encoders[col] = encoder

X = df.drop(target, axis=1)
y = df[target]

feature_names = X.columns.tolist()

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

print("\nAccuracy")
print(accuracy_score(y_test, pred))

print("\nClassification Report")
print(classification_report(y_test, pred))

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(encoders, open("encoders.pkl", "wb"))

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))

HTML = """
<!DOCTYPE html>
<html lang="en">

<head>

<meta charset="UTF-8">

<meta name="viewport"
content="width=device-width, initial-scale=1">

<script src="https://cdn.tailwindcss.com"></script>

<title>AI Obesity Prediction</title>

</head>

<body class="bg-slate-100">

<div class="max-w-6xl mx-auto p-8">

<div class="bg-white rounded-2xl shadow-xl p-8">

<h1 class="text-4xl font-bold text-center text-blue-700 mb-2">
AI Based Obesity Prediction System
</h1>

<p class="text-center text-gray-500 mb-8">
Machine Learning Project using Flask
</p>

<form method="POST" action="/predict">

<div class="grid grid-cols-1 md:grid-cols-2 gap-5">
                <div>
                    <label class="font-semibold">Gender</label>
                    <select name="Gender" class="w-full border rounded-lg p-2 mt-1">
                        <option>Male</option>
                        <option>Female</option>
                    </select>
                </div>

                <div>
                    <label class="font-semibold">Age</label>
                    <input type="number" step="1" name="Age" required class="w-full border rounded-lg p-2 mt-1">
                </div>

                <div>
                    <label class="font-semibold">Height (meters)</label>
                    <input type="number" step="0.01" name="Height" required class="w-full border rounded-lg p-2 mt-1">
                </div>

                <div>
                    <label class="font-semibold">Weight (kg)</label>
                    <input type="number" step="0.1" name="Weight" required class="w-full border rounded-lg p-2 mt-1">
                </div>

                <div>
                    <label class="font-semibold">Family History</label>
                    <select name="family_history_with_overweight" class="w-full border rounded-lg p-2 mt-1">
                        <option>yes</option>
                        <option>no</option>
                    </select>
                </div>

                <div>
                    <label class="font-semibold">High Calorie Food</label>
                    <select name="FAVC" class="w-full border rounded-lg p-2 mt-1">
                        <option>yes</option>
                        <option>no</option>
                    </select>
                </div>

                <div>
                    <label class="font-semibold">Vegetable Consumption (FCVC)</label>
                    <input type="number" step="0.1" name="FCVC" required class="w-full border rounded-lg p-2 mt-1">
                </div>

                <div>
                    <label class="font-semibold">Meals Per Day (NCP)</label>
                    <input type="number" step="0.1" name="NCP" required class="w-full border rounded-lg p-2 mt-1">
                </div>

                <div>
                    <label class="font-semibold">Food Between Meals</label>
                    <select name="CAEC" class="w-full border rounded-lg p-2 mt-1">
                        <option>no</option>
                        <option>Sometimes</option>
                        <option>Frequently</option>
                        <option>Always</option>
                    </select>
                </div>

                <div>
                    <label class="font-semibold">Smoking</label>
                    <select name="SMOKE" class="w-full border rounded-lg p-2 mt-1">
                        <option>yes</option>
                        <option>no</option>
                    </select>
                </div>

                <div>
                    <label class="font-semibold">Water Intake (CH2O)</label>
                    <input type="number" step="0.1" name="CH2O" required class="w-full border rounded-lg p-2 mt-1">
                </div>

                <div>
                    <label class="font-semibold">Calories Monitoring</label>
                    <select name="SCC" class="w-full border rounded-lg p-2 mt-1">
                        <option>yes</option>
                        <option>no</option>
                    </select>
                </div>

                <div>
                    <label class="font-semibold">Physical Activity (FAF)</label>
                    <input type="number" step="0.1" name="FAF" required class="w-full border rounded-lg p-2 mt-1">
                </div>

                <div>
                    <label class="font-semibold">Technology Usage (TUE)</label>
                    <input type="number" step="0.1" name="TUE" required class="w-full border rounded-lg p-2 mt-1">
                </div>

                <div>
                    <label class="font-semibold">Alcohol Consumption</label>
                    <select name="CALC" class="w-full border rounded-lg p-2 mt-1">
                        <option>no</option>
                        <option>Sometimes</option>
                        <option>Frequently</option>
                        <option>Always</option>
                    </select>
                </div>

                <div>
                    <label class="font-semibold">Transportation</label>
                    <select name="MTRANS" class="w-full border rounded-lg p-2 mt-1">
                        <option>Walking</option>
                        <option>Bike</option>
                        <option>Motorbike</option>
                        <option>Public_Transportation</option>
                        <option>Automobile</option>
                    </select>
                </div>

            </div>

            <div class="text-center mt-8">

                <button
                class="bg-blue-600 hover:bg-blue-700 text-white px-8 py-3 rounded-xl text-lg">
                Predict Obesity
                </button>

            </div>

            {% if prediction %}

<div class="mt-8 bg-green-100 rounded-xl p-6 shadow">

<h2 class="text-3xl font-bold text-green-700 mb-4">
Prediction : {{ prediction }}
</h2>

<p class="text-xl">
<b>BMI :</b> {{ bmi }}
</p>

<p class="text-xl mt-2">
<b>BMI Status :</b> {{ bmi_status }}
</p>

<p class="text-lg mt-4">
<b>Recommendation :</b><br>
{{ recommendation }}
</p>

</div>

{% endif %}

        </form>

    </div>

</div>

<script>

document.querySelector("form").addEventListener("submit",function(){

document.querySelector("button").innerHTML="Predicting...";

});

</script>

</body>

</html>
"""

@app.route("/")
def home():
    return render_template_string(HTML)

@app.route("/predict", methods=["POST"])
def predict():

    values = []

    for column in feature_names:

        value = request.form[column]

        if column in encoders:
            value = encoders[column].transform([value])[0]
        else:
            value = float(value)

        values.append(value)
    input_df = pd.DataFrame([values], columns=feature_names)

    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]

    prediction = encoders["NObeyesdad"].inverse_transform([prediction])[0]

    height = float(request.form["Height"])
    weight = float(request.form["Weight"])

    bmi = weight / (height * height)

    if bmi < 18.5:
        bmi_status = "Underweight"
        recommendation = "Increase healthy calorie intake, eat protein-rich foods, and consult a nutritionist if necessary."

    elif bmi < 25:
        bmi_status = "Normal Weight"
        recommendation = "Maintain your healthy lifestyle with balanced nutrition and regular exercise."

    elif bmi < 30:
        bmi_status = "Overweight"
        recommendation = "Exercise at least 30 minutes daily, reduce sugary foods, and increase vegetables."

    else:
        bmi_status = "Obese"
        recommendation = "Consult a healthcare professional, follow a structured diet plan, and perform regular physical activity."

    return render_template_string(
        HTML,
        prediction=prediction,
        bmi=round(bmi, 2),
        bmi_status=bmi_status,
        recommendation=recommendation
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

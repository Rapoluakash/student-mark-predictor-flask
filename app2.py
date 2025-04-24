# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

# ✅ Load the model (file path must be correct)
model = joblib.load("student_mark_predictor.pkl")


# DataFrame to store input + prediction
df = pd.DataFrame()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global df

    input_features = [int(x) for x in request.form.values()]
    features_value = np.array(input_features)

    # ✅ Validate: hours should be between 1 and 24
    if input_features[0] < 1 or input_features[0] > 24:
        return render_template(
            'index.html',
            prediction_text='❌ Please enter valid hours between 1 to 24 if you live on the Earth.'
        )

    # ✅ Predict marks
    output = model.predict([features_value])[0][0].round(2)

    # ✅ Store input and output in CSV
    df = pd.concat([
        df,
        pd.DataFrame({'Study Hours': input_features, 'Predicted Output': [output]})
    ], ignore_index=True)

    df.to_csv('smp_data_from_app.csv', index=False)

    return render_template(
        'index.html',
        prediction_text=f'✅ You will get [{output}%] marks when you study [{int(features_value[0])}] hours per day.'
    )

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000, debug=True, use_reloader=False)


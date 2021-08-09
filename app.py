import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import joblib
import autoai_libs

app = Flask(__name__) #Initialize the flask App
#model = pickle.load(open('model.pkl', 'rb'))
model = joblib.load('P4.pickle')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    features = [x for x in request.form.values()]
    final_features = np.array([features])
    prediction = model.predict(final_features)

    #output = round(prediction[0], 2)

    #return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(output))
    return render_template('index.html', prediction_text='Churn Classification should be {}'.format(prediction))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
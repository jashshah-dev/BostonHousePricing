import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd
#Starting point
app=Flask(__name__)
#load the model
linearRegressionModel=pickle.load(open('linearRegressionModel.pkl','rb'))
scaling=pickle.load(open('scaling.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])

def predict_api():
    #input in JSON request
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data=scaling.transform(np.array(list(data.values())).reshape(1,-1))
    output=linearRegressionModel.predict(new_data)
    print(output[0])
    return jsonify(output[0])

if __name__=="__main__":
    app.run(debug=True)

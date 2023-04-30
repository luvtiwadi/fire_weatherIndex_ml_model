from flask import Flask
from flask import render_template ,request,jsonify
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
app = Flask(__name__)


@app.route('/')
def start():
    return render_template("index.html")

regressor=pickle.load(open('models/regressor12.pkl','rb'))
scaler=pickle.load(open('models/scaler12.pkl','rb'))

@app.route('/predict',methods=['POST'])
def calculations():
    if request.method == 'POST':
        temp=float(request.form['temperature'])
        rh=float(request.form['rh'])
        ws=float(request.form['ws'])
        rain=float(request.form['rain'])
        ffmc=float(request.form['ffmc'])
        dmc=float(request.form['dmc'])
        dc=float(request.form['dc'])
        isi=float(request.form['isi'])
        bui=float(request.form['bui'])
        classes=float(request.form['classes'])
        region=float(request.form['region'])
        new_data=scaler.transform([[temp,rh,ws,rain,ffmc,dmc,dc,isi,bui,classes,region]])
        result1=regressor.predict(new_data)

        
        return render_template('index.html',result=result1[0])


if __name__=="__main__":
    app.run(host="0.0.0.0")
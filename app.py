import numpy as np
from flask import Flask, render_template,request
import pandas as pd
import pickle#Initialize the flask App
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

stn_map = {'TG001':1, 'TG002':2, 'TG003':3, 'TG004':4, 'TG006':6, 'AP005':5}
def conv_input(date_in, sid):
    date_in = date_in.toordinal()
    sid = stn_map[sid]
    return [sid,date_in]

@app.route('/predict',methods=['POST'])
def predict():
    #For rendering results on HTML GUI
    # int_features = [float(x) for x in request.form.values()]
    # final_features = [np.array(int_features)]
    input_stn = request.form["station_id"]
    input_date = request.form["date"]
    X_in = conv_input(pd.Timestamp(input_date),input_stn)
    final_X = [np.array(X_in)]
    prediction = model.predict(final_X)
    output = round(prediction[0][0], 2) 
    return render_template('index.html', prediction_text='AQI predicted is :{}'.format(output))



if __name__ == "__main__":
    app.run(debug=True)
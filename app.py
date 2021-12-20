from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

df = pd.read_excel('FullMatrixAllAlloy.xlsx', index_col='Alloy')

model = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)



@app.route('/')
def main():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def home():
    data1 = request.form['a']
    data2 = request.form['b']
    data3 = request.form['c']
    data4 = request.form['d']
    data5 = request.form['e']
    data6 = request.form['f']
    data7 = request.form['g']
    arr = np.array([[data1, data2, data3, data4, data5, data6, data7]])
    pred = model.predict(arr)
    return render_template('result.html',
                           HT = data1,
                           Si = data2,
                           Cu = data3,
                           Mn = data4,
                           Mg = data5,
                           Cr = data6,
                           Zr = data7, data=pred[:,0], data0=pred[:,1])



if __name__ == "__main__":
    app.run(debug=True)

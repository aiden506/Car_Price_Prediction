import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle


app = Flask(__name__)
path=r'C:\\Users\\AIDEN SAMUEL\\PycharmProjects\\Pythonproject1\model_car3.pkl'
model = pickle.load(open(path, 'rb'))
#Car pred

@app.route('/')
def home():
    return render_template('Carpred.xhtml')
    # return render_template('index.html')

@app.route('/Carpred', methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    # output=prediction[0]
    output = round(prediction[0], 2) *100
    # return render_template('Carpred.html')
    return render_template('Carpred.html', prediction_text='The best price predicted for this car : {}'.format(output))

@app.route('/AboutUs')
def about():
    return render_template('AboutUs.html')

@app.route('/createAccount')
def createAcc():
    return render_template('createAccount.html')



if __name__ == "__main__":
    app.run(debug=True)
import pickle
from flask import Flask, request, app, jsonify, render_template

app = Flask(__name__)
# Load the model 
text_clf_model = pickle.load(open('text_clf_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    output = text_clf_model.predict([data])
    return jsonify(int(output[0]))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.get("TEXT")
    output = text_clf_model.predict([data])
    return render_template("home.html", prediction_text = "The predicted class is {}.".format(output[0]))

if __name__=="__main__":
    app.run(debug=True)
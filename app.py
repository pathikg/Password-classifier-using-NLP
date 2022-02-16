from flask import Flask, render_template, request
from joblib import load
import xgboost

app = Flask(__name__)

dick = {0: 'weak', 1: 'moderate', 2: "strong"}


def word_char(inputs):
    a = []
    for i in inputs:
        a.append(i)
    return a


xgboost_dict = load(open('xgb_pipeline_objects.pkl', 'rb'))


def predict_label(password):
    vect = xgboost_dict['vectorizer']
    model = xgboost_dict['model']
    inp = vect.transform([password])
    pred = model.predict(inp)
    return dick[pred[0]]

# routes


@app.route("/", methods=['GET', 'POST'])
def kuch_bhi():
    return render_template("home.html")


@app.route("/about")
def about_page():
    return "About You..!!!"


@app.route("/submit", methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        password = request.form['pwd']
        p = predict_label(password)

    return render_template("home.html", prediction=p)

if __name__ == '__main__':
    #app.debug = True

    app.run(debug=True)

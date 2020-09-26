import numpy as np
import pickle
from flask import Flask, Response, request, render_template, jsonify

app = Flask('my_app')
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/form')
def form():
    return render_template('form.html')

@app.route('/submit')
def submit():
    true_count = 0
    data = request.args
    your_news = data['NewsInput']
    X_test = np.array([data['NewsInput']])

    cvl = pickle.load(open('./pfiles/Count_Vectorizer_and_Logistic_Regression_CV.p', 'rb'))
    cvl_pred = cvl.predict(X_test)
    cvl_pred = cvl_pred[0]
    if cvl_pred == 'true':
        true_count += 1

    cvmnb = pickle.load(open('./pfiles/Count_Vectorizer_and_Multinomial_Naive_Bayes.p', 'rb'))
    cvmnb_pred = cvmnb.predict(X_test)
    cvmnb_pred = cvmnb_pred[0]
    if cvmnb_pred == 'true':
        true_count += 1

    cvrf = pickle.load(open('./pfiles/Count_Vectorizer_and_Random_Forest.p', 'rb'))
    cvrf_pred = cvrf.predict(X_test)
    cvrf_pred = cvrf_pred[0]
    if cvrf_pred == 'true':
        true_count += 1

    tfl = pickle.load(open('./pfiles/TF-IDF_Vectorizer_and_Logistic_Regression_CV.p', 'rb'))
    tfl_pred = tfl.predict(X_test)
    tfl_pred = tfl_pred[0]
    if tfl_pred == 'true':
        true_count += 1

    tfmnb = pickle.load(open('./pfiles/TF-IDF_Vectorizer_and_Multinomial_Naive_Bayes.p', 'rb'))
    tfmnb_pred = tfmnb.predict(X_test)
    tfmnb_pred = tfmnb_pred[0]
    if tfmnb_pred == 'true':
        true_count += 1

    tfrf = pickle.load(open('./pfiles/TF-IDF_Vectorizer_and_Random_Forest.p', 'rb'))
    tfrf_pred = tfrf.predict(X_test)
    tfrf_pred = tfrf_pred[0]
    if tfrf_pred == 'true':
        true_count += 1

    if true_count >= 3:
        con = 'More than half of the models predict your news is TRUE!' \
              ' Please keep in mind this app is not 100% accurate, continue practicing social distancing, and wearing your mask. Please use common sense when judging the news and instructions you read. Always do what is safest for you and the people around you.'
    else:
        con = 'More than half of the models predict your news is FALSE. Please check your source!!!'\
              ' Please keep in mind this app is not 100% accurate, continue practicing social distancing, and wearing your mask. Please use common sense when judging the news and instructions you read. Always do what is safest for you and the people around you.'


    return render_template('result.html', news = your_news, cvl_prediction = cvl_pred, cvmnb_prediction = cvmnb_pred, cvrf_prediction = cvrf_pred, tfl_prediction = tfl_pred, tfmnb_prediction = tfmnb_pred, tfrf_prediction = tfrf_pred, conclusion = con )

if __name__ =='__main__': #if weS run python app_starter.py in the terminal
    app.run(debug = True)

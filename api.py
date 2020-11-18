from flask import Flask, jsonify, request
#[1] importo o desrializador
from sklearn.externals import  joblib


#[2] Carrego a classe de predição do diretório local
clf = joblib.load('decision_tree.pk1')
app =  Flask(__name__)
@app.route('/flowers_predictor')
def flowers_predictor():

    sepal_length = float(request.args.get('sepal_length'))
    sepal_width = float(request.args.get('sepal_width'))
    petal_length = float(request.args.get('petal_length'))
    petal_width = float(request.args.get('petal_width'))

    event = [sepal_width, sepal_width, petal_length, petal_width]
    target_names = ['Setosa', 'Versicolor', 'Virginica']

    result = {}

    # [4] Realiza predição com base no evento
    prediction = clf.predict([event])[0]

    # [5] Realizar probabilidades individuais das três classes
    probas = list(zip(target_names, clf.predict_proba([event])[0]))

    # [6] Recupera o nome real da classe
    result['prediction'] = target_names[prediction]
    result['probas'] = probas

    return jsonify(result), 200

app.run(debug=True, use_reloader=True)
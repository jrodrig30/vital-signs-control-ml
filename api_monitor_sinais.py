from flask import Flask, jsonify, request
#[1] importo o desrializador
from sklearn.externals import  joblib


#[2] Carrego a classe de predição do diretório local
clf = joblib.load('decision_tree.pk1')
app =  Flask(__name__)
@app.route('/get_diagnostico')
def get_diagnostico():
    idade = int(request.args.get('idade'))
    frequencia_cardiaca = int(request.args.get('frequencia_cardiaca'))
    frequencia_respiratoria = int(request.args.get('frequencia_respiratoria'))
    pressao_dia = int(request.args.get('pressao_dia'))
    pressao_sis = int(request.args.get('pressao_sis'))
    pam =float(request.args.get('pam'))
    saturacao_oxigenio = int(request.args.get('saturacao'))
    temperatura = float(request.args.get('temperatura'))

    event = [idade, frequencia_cardiaca, frequencia_respiratoria, pressao_dia, pressao_sis, pam, saturacao_oxigenio, temperatura]
    #target_names = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]

    result = {}

    # [4] Realiza predição com base no evento
    prediction = clf.predict([event])[0]

    # [5] Realizar probabilidades individuais das três classes
    #probas = list(zip(target_names, clf.predict_proba([event])[0]))

    # [6] Recupera o nome real da classe
    result['prediction'] = str(prediction)
    result['erro'] = 'sem'

    return jsonify(result), 200

app.run(debug=True, use_reloader=True)

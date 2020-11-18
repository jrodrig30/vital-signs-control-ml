import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import os


sinais = pd.read_csv(os.path.abspath('/media/jose/7a28a73a-96b5-48c8-bb84-dd001dfd0fb5/www/physionet/public_html/python/data-set-sinais-vitais.csv'), delimiter=',')
atributos = list(sinais.columns[:8])
classe =  list(sinais.columns[8:9])
atributos = sinais.as_matrix(atributos)
classe = sinais.as_matrix(classe)
atributos_treino, atributos_teste, classe_treino, classe_teste = train_test_split(atributos, classe)
classificador = DecisionTreeClassifier()
classificador.fit(atributos_treino, classe_treino)
from sklearn.externals import joblib
joblib.dump(classificador, os.path.abspath('/media/jose/7a28a73a-96b5-48c8-bb84-dd001dfd0fb5/www/physionet/public_html/python/decision_tree.pk1'))
#teste_predict = classificador.predict(atributos_teste)
#print(classificador.predict([[29,101,15,80,120,130,99,38]]))
#print ("A taxa de acerto é de ")
#print(accuracy_score(classe_teste, teste_predict)*100)

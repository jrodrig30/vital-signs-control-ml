import pandas as pd
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


sinais = pd.read_csv(os.path.abspath("/var/www/physionet/public_html/python/data-set-sinais-vitais.csv"), delimiter=',')
print('Quantidade de registros totais: ')
print(len(sinais))

#print(X)
atributos = list(sinais.columns[:8]) #São 8 atributos
classe =  list(sinais.columns[8:9]) #Uma unica classe
atributos = sinais.as_matrix(atributos)
classe = sinais.as_matrix(classe)

atributos_treino, atributos_teste, classe_treino, classe_teste = train_test_split(atributos, classe, test_size=0.3,random_state=100)

print('Quantidade atributos teste: ')
print(len(atributos_teste))

classificador = DecisionTreeClassifier(criterion='gini')
classificador.fit(atributos_treino, classe_treino)
teste_predict = classificador.predict(atributos_teste)

print ("A taxa de acerto é de: ")
print(accuracy_score(classe_teste, teste_predict)*100)

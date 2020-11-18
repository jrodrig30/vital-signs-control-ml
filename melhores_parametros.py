import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit


sinais = pd.read_csv(os.path.abspath("/var/www/physionet/public_html/python/data-set-sinais-vitais.csv"), delimiter=',')
print('Quantidade de registros totais: ')
print(len(sinais))

features = sinais.columns.difference(['classe'])

X = sinais[features].values
y = sinais['classe'].values
print('Quantidade de classes: ')
print(len(sinais['classe'].value_counts()))

print('Quantidade por classe: ')
print(sinais['classe'].value_counts())

'''
scikit-learn usa uma versão otimizada do algoritmo CART
Parâmetros DecisionTreeClassifier
random_state: É comum na maioria dos algoritmos e é importante
              mantê-lo fixo, o valor não importa, desde que seja sempre o mesmo,
              dessa forma conseguiremos gerar sempre o mesmo modelo com os mesmos dados.
              
criterion: É a métrica utilizada para construção da árvore de decisão. Pode ser gini ou entropy.

max_depth: É a profundida máxima da árvore, profundida demais pode gerar um sistema super
           especializado nos dados de treinamento, também conhecido como overfitting.
           Profundida de menos vai diminuir a capacidade de generalização do modelo.
'''


classificador = DecisionTreeClassifier()
classificador.fit(X, y)  # Treinando com tudo

features_importance = zip(classificador.feature_importances_, features)
for importance, feature in sorted(features_importance, reverse=True):
   print("%s: %f%%" % (feature, importance*100))



from sklearn.model_selection import GridSearchCV
param_grid = {
            "criterion": ['entropy', 'gini']
}
grid_search = GridSearchCV(classificador, param_grid, scoring="accuracy")
grid_search.fit(X, y)

classificador = grid_search.best_estimator_
grid_search.best_params_, grid_search.best_score_

print(grid_search.best_score_)
print(grid_search.best_params_)

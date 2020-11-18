import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import cross_val_score

sinais = pd.read_csv(os.path.abspath("/var/www/physionet/public_html/python/data-set-sinais-vitais.csv"), delimiter=',')
print('Quantidade de registros totais: ')
print(len(sinais))
features = sinais.columns.difference(['classe'])

X = sinais[features].values
y = sinais['classe'].values

classificador = DecisionTreeClassifier()
classificador.fit(X, y)  # Treinando com tudo

scores_dt = cross_val_score(classificador, X, y,scoring='accuracy', cv=5)

print('A taxa de acerto do modelo é: ')
print(scores_dt.mean() * 100);

import time
inicio = time.time()
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import graphviz
import csv

def holdout:

    
data = []
with open('diabetes_indicator.csv', mode='r') as file:
    csv_reader = csv.DictReader(file)
    for row in csv_reader:
        data.append(row)

df = pd.DataFrame(data)
print(df)
'''X = df[['Diabetes_012']]  # Usando [[]] para criar um DataFrame com uma coluna
y = df['HighBP']

modelov1 = DecisionTreeClassifier(max_depth=None,
                                  max_features=None,
                                  criterion="entropy",
                                  min_samples_leaf=1,
                                  min_samples_split=2)
modelov1.fit(X, y)

arquivo = 'tree_modelov1.dot'

export_graphviz(modelov1, out_file=arquivo, feature_names=['Diabetes_012'])
with open(arquivo) as f:
    dot_graph = f.read()
grafico = graphviz.Source(dot_graph)
grafico.view()'''

fim = time.time()
print(f"{fim - inicio} s")

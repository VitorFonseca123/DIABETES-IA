import time

import graphviz
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier, export_graphviz

inicio = time.time()

def renomear(df):
    new_column_names = {
        0: 'HighBP', 1: 'HighChol', 2: 'CholCheck', 3: 'BMI', 4: 'Smoker',
        5: 'Stroke', 6: 'HeartDiseaseorAttack', 7: 'PhysActivity', 8: 'Fruits',
        9: 'Veggies', 10: 'HvyAlcoholConsump', 11: 'AnyHealthcare', 12: 'NoDocbcCost',
        13: 'GenHlth', 14: 'MentHlth', 15: 'PhysHlth', 16: 'DiffWalk', 17: 'Sex',
        18: 'Age', 19: 'Education', 20: 'Income'
    }
    # renomeia o dataframe
    df.rename(columns=new_column_names, inplace=True)
    return df

def holdout(df):
    X = df.iloc[:, 1:].values
    y = df.iloc[:, 0].values

    # Dividindo os dados em treino e teste (80% treino, 20% teste)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

def r_fold_cross_validation(df, n_splits):
    X = df.iloc[:, 1:].values
    y = df.iloc[:, 0].values

    kfold = KFold(n_splits, shuffle=True, random_state=42)
    for i, (train_index, test_index) in enumerate(kfold.split(X)):
        '''print(f"Fold {i}:")
        print(f"  Train: index={train_index}")
        print(f"  Test:  index={test_index}")'''

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        return X_train, X_test, y_train, y_test

#ler arquivo csv
df = pd.read_csv('diabetes_indicator.csv')

#X_train, X_test, y_train, y_test = holdout(df)
X_train, X_test, y_train, y_test = r_fold_cross_validation(df,3)
treinamento = pd.DataFrame(X_train)
teste = pd.DataFrame(X_test)

teste = renomear(teste)
treinamento = renomear(treinamento)
# Obter os nomes das colunas (exceto a primeira, que é a etiqueta)
feature_names = df.columns[1:]

'''
print(f"Número de linhas de treinamento: {len(treinamento)}")
print(f"Número de linhas de teste: {len(teste)}")'''

print(treinamento)
print(teste)

arvore_decisao = DecisionTreeClassifier(max_depth=None,
                                  max_features=None,
                                  criterion="entropy",
                                  min_samples_leaf=1,
                                  min_samples_split=2)
arvore_decisao.fit(X_train, y_train)
arquivo = 'tree_modelov1.dot'


export_graphviz(arvore_decisao, out_file=arquivo, feature_names=feature_names, class_names=True)
with open(arquivo) as f:
    dot_graph = f.read()
grafico = graphviz.Source(dot_graph)
grafico.render(view=True)




fim = time.time()
print(f"{fim - inicio:.2f} s")

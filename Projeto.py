import time

import graphviz
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
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

def holdout(df, test_size):
    X = df.iloc[:, 1:].values
    y = df.iloc[:, 0].values

    # Dividindo os dados em treino e teste (80% treino, 20% teste)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)


    treinamento = pd.DataFrame(X_train)
    teste = pd.DataFrame(X_test)

    teste = renomear(teste)
    treinamento = renomear(treinamento)
    # Obter os nomes das colunas (exceto a primeira, que é a etiqueta)
    feature_names = df.columns[1:]
    print("Accuracy")
    print("Arvore: ", arvore_decisao(treinamento, y_train, feature_names, teste, y_test))
    print("KNN: ",KNN(treinamento, y_train, 10, teste, y_test))


def r_fold_cross_validation(df, r_folds):
    X = df.iloc[:, 1:].values
    y = df.iloc[:, 0].values

    kfold = KFold(r_folds, shuffle=True, random_state=1)
    accuracies_tree = []
    accuracies_KNN = []
    for i, (train_index, test_index) in enumerate(kfold.split(X)):
        '''print(f"Fold {i}:")
        print(f"  Train: index={train_index}")
        print(f"  Test:  index={test_index}")'''

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        treinamento = pd.DataFrame(X_train)
        teste = pd.DataFrame(X_test)

        teste = renomear(teste)
        treinamento = renomear(treinamento)
        # Obter os nomes das colunas (exceto a primeira, que é a etiqueta)
        feature_names = df.columns[1:]
        accuracies_tree.append(arvore_decisao(treinamento, y_train, feature_names, teste, y_test))
        accuracies_KNN.append(KNN(treinamento, y_train, 10, teste, y_test))
    print("Accuracy")
    print("Arvore: ", np.mean(accuracies_tree))
    print("KNN: ", np.mean(accuracies_KNN))


def arvore_decisao(treinamento, y_train, f_names,x_teste, y_teste):
    arvore_decisao = DecisionTreeClassifier(max_depth=10,
                                            max_features=None,
                                            criterion="entropy",
                                            min_samples_leaf=1,
                                            min_samples_split=2)
    arvore_decisao.fit(treinamento, y_train)
    '''arquivo = 'tree_modelov1.dot'

    export_graphviz(arvore_decisao, out_file=arquivo, feature_names=f_names, class_names=True)
    with open(arquivo) as f:
        dot_graph = f.read()
    grafico = graphviz.Source(dot_graph)
    grafico.render(view=True)'''

    #print(arvore_decisao.get_depth())
    #print(arvore_decisao.get_n_leaves())
    y_pred = arvore_decisao.predict(x_teste)
    #print(accuracy_score(y_teste,y_pred))
    #print(arvore_decisao.score(x_teste, y_teste, None))
    return accuracy_score(y_teste,y_pred)

def KNN(treinamento,y_train,K, x_teste, y_teste):
    k_vizinhos = KNeighborsClassifier(n_neighbors=K,
                                      weights='uniform',
                                      algorithm='auto',
                                      p=2,
                                      metric_params=None,
                                      n_jobs=None)
    k_vizinhos.fit(treinamento, y_train)

    y_pred = k_vizinhos.predict(x_teste)
    return accuracy_score(y_teste, y_pred)

def main():
    # ler arquivo csv
    df = pd.read_csv('diabetes_indicator.csv')



    '''
    print(f"Número de linhas de treinamento: {len(treinamento)}")
    print(f"Número de linhas de teste: {len(teste)}")'''
    #print(df)
    #print(treinamento)
    #print(teste)

    #holdout(df,0.2)
    r_fold_cross_validation(df,2)
    fim = time.time()
    print(f"{fim - inicio:.2f} s")

main()
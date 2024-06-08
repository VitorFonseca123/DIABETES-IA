import time
from imblearn.over_sampling import SMOTE
import graphviz
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import plotly.express as px


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


def holdout(df, test_size, f_names, X, y, modelo, modelo_nome):


    # Dividindo os dados em treino e teste (80% treino, 20% teste)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,shuffle=True)



    count0, count1, count2 = sum(y_train == 0), sum(y_train == 1), sum(y_train == 2)
    print(count0)
    print(count1)
    print(count2)
    print("----")
    # Determinar o fator de aumento para cada classe


    smote = SMOTE(sampling_strategy='not majority',random_state=65478798)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    count0, count1, count2 = sum(y_train_resampled == 0), sum(y_train_resampled == 1), sum(y_train_resampled == 2)
    print(count0)
    print(count1)
    print(count2)
    treinamento = pd.DataFrame(X_train_resampled)



    treinamento = renomear(treinamento)

    teste = pd.DataFrame(X_test)
    teste = renomear(teste)
    print("Holdout")
    modelo.fit(treinamento, y_train_resampled)
    if modelo_nome.upper() == 'ARVORE':
        arquivo = 'tree_modelov1.dot'

        export_graphviz(modelo, out_file=arquivo, feature_names=f_names, class_names=True)
        with open(arquivo) as f:
            dot_graph = f.read()
        grafico = graphviz.Source(dot_graph)
        grafico.render(view=True)
        y_pred = modelo.predict(teste)
        print(metrics.confusion_matrix(y_test, y_pred))
        print("Acuracia:", metrics.accuracy_score(y_test, y_pred))
        print("Precisao: ", metrics.precision_score(y_test, y_pred, average=None))
    elif modelo_nome.upper() == 'KNN':
        pca = PCA(n_components=2)
        principalComponents_train = pca.fit_transform(X_train_resampled)
        principalComponents_test = pca.transform(X_test)

        treinamento = pd.DataFrame(principalComponents_train,
                                   columns=['componente principal 1', 'componente principal 2'])
        teste = pd.DataFrame(principalComponents_test, columns=['componente principal 1', 'componente principal 2'])

        print("KNN: ")
        modelo.fit(treinamento, y_train_resampled)
        y_pred = modelo.predict(teste)

        print(metrics.confusion_matrix(y_test, y_pred))
        print("Acuracia: ", metrics.accuracy_score(y_test, y_pred))
        print("Precisao: ", metrics.precision_score(y_test, y_pred, average=None))

        y_score = modelo.predict_proba(teste)[:, 1]
        fig = px.scatter(
            principalComponents_test, x=0, y=1,
            color=y_score, color_continuous_scale='RdBu',
            symbol=y_test, symbol_map={'0': 'square-dot', '1': 'circle-dot'},
            labels={'symbol': 'label', 'color': 'score of <br>first class'}
        )
        fig.update_traces(marker_size=12, marker_line_width=1.5)
        fig.update_layout(legend_orientation='h')
        fig.show()


def r_fold_cross_validation(df, r_folds,modelo, modeloNome):
    X = df.iloc[:, 1:].values
    y = df.iloc[:, 0].values
    acc = []
    prec = []
    kf = StratifiedKFold(n_splits=r_folds, shuffle=True, random_state=42)
    for i, (train_index, test_index) in enumerate(kf.split(X, y)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        smote = SMOTE(sampling_strategy='minority',random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

        treinamento = pd.DataFrame(X_train_resampled)
        teste = pd.DataFrame(X_test)

        teste = renomear(teste)
        treinamento = renomear(treinamento)

        modelo.fit(treinamento, y_train_resampled)
        y_pred = modelo.predict(teste)
        print(metrics.confusion_matrix(y_test, y_pred))
        acc.append(metrics.accuracy_score(y_test, y_pred))
        prec.append(metrics.precision_score(y_test,y_pred, average=None))
    mediAcc = np.mean(acc)
    mediaPrec_classes = np.mean(prec, axis=0)
    print("Acuracia: ", mediAcc)
    print("Precisao: ", mediaPrec_classes)


def arvore_decisao(profundidade, minLeaf, minSample):
    arvore_decisao = DecisionTreeClassifier(max_depth=profundidade,
                                            max_features=10,
                                            criterion="entropy",
                                            min_samples_leaf=minLeaf,
                                            min_samples_split=minSample)




    return arvore_decisao

def KNN(K):
    k_vizinhos = KNeighborsClassifier(n_neighbors=K,
                                      weights='uniform',
                                      algorithm='auto',
                                      p=2,
                                      metric_params=None,
                                      n_jobs=None)

    return k_vizinhos

def main():
    inicio = time.time()
    # ler arquivo csv
    df = pd.read_csv('diabetes_indicator.csv')
    #primeira_feature = df[df.iloc[:, 0] == 0]

    # Selecionar metade desses registros de forma aleatória
    #metade_para_remover = primeira_feature.sample(frac=0.5, random_state=1)

    # Remover esses registros do DataFrame original
    #df = df.drop(metade_para_remover.index)

    primeira_feature = df[df.iloc[:, 0] == 1]
    metade_para_remover = primeira_feature.sample(frac=0, random_state=1)
    df = df.drop(metade_para_remover.index)

    '''
    print(f"Número de linhas de treinamento: {len(treinamento)}")
    print(f"Número de linhas de teste: {len(teste)}")'''
    #print(df)
    #print(treinamento)
    #print(teste)
    feature_names = df.columns[1:]
    X = df.iloc[:, 1:].values
    y = df.iloc[:, 0].values
    AD = arvore_decisao(10,1,2)
    vizinhos = KNN(100)
    df.sample()
    holdout(df, 0.2, feature_names, X, y, AD, 'Arvore')
    holdout(df,0.2, feature_names,X, y, vizinhos, 'KNN')
    #r_fold_cross_validation(df,3, AD, 'Arvore')
    #r_fold_cross_validation(df, 3, vizinhos, 'KNN')
    fim = time.time()
    print(f"{fim - inicio:.2f} s")

main()
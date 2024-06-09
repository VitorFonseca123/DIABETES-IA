import time
from imblearn.over_sampling import SMOTE
import graphviz
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import plotly.express as px
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


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


def plot_roc_curve(y_test, y_pred_proba, model_name):
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1], pos_label=1)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic - {model_name}')
    plt.legend(loc="lower right")
    plt.show()


def holdout(df, test_size, f_names, X, y, modelo, modelo_nome):
    # Dividindo os dados em treino e teste (80% treino, 20% teste)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=True)

    treinamento = pd.DataFrame(X_train)
    treinamento = renomear(treinamento)

    teste = pd.DataFrame(X_test)
    teste = renomear(teste)
    print("Holdout")
    modelo.fit(treinamento, y_train)

    if modelo_nome.upper() == 'ARVORE':
        arquivo = 'tree_modelov1.dot'
        print("Arvore")
        export_graphviz(modelo, out_file=arquivo, feature_names=f_names, class_names=True)
        with open(arquivo) as f:
            dot_graph = f.read()
        grafico = graphviz.Source(dot_graph)
        grafico.render(view=True)
        y_pred = modelo.predict(teste)
        y_pred_proba = modelo.predict_proba(teste)
        print(metrics.confusion_matrix(y_test, y_pred))
        print("Acuracia:", metrics.accuracy_score(y_test, y_pred))
        print("Precisao: ", metrics.precision_score(y_test, y_pred, average=None))
        plot_roc_curve(y_test, y_pred_proba, 'Decision Tree')

    elif modelo_nome.upper() == 'KNN':
        print("KNN: ")
        modelo.fit(treinamento, y_train)
        y_pred = modelo.predict(teste)
        y_pred_proba = modelo.predict_proba(teste)
        print(metrics.confusion_matrix(y_test, y_pred))
        print("Acuracia: ", metrics.accuracy_score(y_test, y_pred))
        print("Precisao: ", metrics.precision_score(y_test, y_pred, average=None))
        plot_roc_curve(y_test, y_pred_proba, 'KNN')

        pca = PCA(n_components=2)
        teste = pca.fit_transform(teste)
        teste = pd.DataFrame(teste, columns=['componente principal 1', 'componente principal 2'])

        fig = px.scatter(teste, x='componente principal 1', y='componente principal 2', color=y_pred)
        fig.update_traces(marker_size=12, marker_line_width=1.5)
        fig.update_layout(legend_orientation='h')
        fig.show()


def r_fold_cross_validation(df, r_folds, modelo, modeloNome):
    X = df.iloc[:, 1:].values
    y = df.iloc[:, 0].values
    acc = []
    prec = []
    kf = StratifiedKFold(n_splits=r_folds, shuffle=True, random_state=42)
    for i, (train_index, test_index) in enumerate(kf.split(X, y)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        smote = SMOTE(sampling_strategy='minority', random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

        treinamento = pd.DataFrame(X_train_resampled)
        teste = pd.DataFrame(X_test)

        teste = renomear(teste)
        treinamento = renomear(treinamento)

        modelo.fit(treinamento, y_train_resampled)
        y_pred = modelo.predict(teste)
        print(metrics.confusion_matrix(y_test, y_pred))
        acc.append(metrics.accuracy_score(y_test, y_pred))
        prec.append(metrics.precision_score(y_test, y_pred, average=None))
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

    primeira_feature = df[df.iloc[:, 0] == 1]
    metade_para_remover = primeira_feature.sample(frac=0, random_state=1)
    df = df.drop(metade_para_remover.index)

    feature_names = df.columns[1:]
    X = df.iloc[:, 1:].values
    y = df.iloc[:, 0].values
    AD = arvore_decisao(10, 1, 2)
    vizinhos = KNN(30)
    df.sample()
    holdout(df, 0.4, feature_names, X, y, AD, 'Arvore')
    holdout(df, 0.4, feature_names, X, y, vizinhos, 'KNN')
    # r_fold_cross_validation(df, 3, AD, 'Arvore')
    # r_fold_cross_validation(df, 3, vizinhos, 'KNN')
    fim = time.time()
    print(f"{fim - inicio:.2f} s")


main()
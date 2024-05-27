import time


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
inicio = time.time()

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
'''
print(f"Número de linhas de treinamento: {len(treinamento)}")
print(f"Número de linhas de teste: {len(teste)}")'''
print(treinamento)
print(teste)





fim = time.time()
print(f"{fim - inicio:.2f} s")

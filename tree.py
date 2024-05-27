import time
import pandas as pd
from sklearn.model_selection import train_test_split

inicio = time.time()

def holdout(df):
    X = df.iloc[:, 1:].values
    y = df.iloc[:, 0].values

    # Dividindo os dados em treino e teste (80% treino, 20% teste)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


#ler arquivo csv
df = pd.read_csv('diabetes_indicator.csv')

# Verificando se há linhas vazias ou com problemas
#print("Número total de linhas no DataFrame:", len(df))
#print("Número de valores faltantes:", df.isnull().sum().sum())

# Removendo linhas com valores faltantes, se existirem
#df = df.dropna()


X_train, X_test, y_train, y_test = holdout(df)
treinamento = pd.DataFrame(X_train)
teste = pd.DataFrame(X_test)

print(f"Número de linhas de treinamento: {len(treinamento)}")
print(f"Número de linhas de teste: {len(teste)}")
print(treinamento)
print(teste)
fim = time.time()
print(f"{fim - inicio:.2f} s")

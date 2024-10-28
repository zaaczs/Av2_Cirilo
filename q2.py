import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.covariance import ShrunkCovariance, LedoitWolf

# Carregar os dados do arquivo
data = pd.read_csv("EMGsDataset.csv", header=None)
X = data.iloc[:2, :].T.values  # Sensores (Corrugador e Zigomático)
y = data.iloc[2, :].values      # Classe

# 1. Visualização inicial dos dados
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis", alpha=0.5)
plt.xlabel("Sensor 1 (Corrugador do Supercílio)")
plt.ylabel("Sensor 2 (Zigomático Maior)")
plt.title("Gráfico de Dispersão dos Dados EMG por Categoria")
plt.colorbar(label="Classe")
plt.show()

# Parâmetros
lambdas = [0, 0.25, 0.5, 0.75]  # valores de lambda para regularização do classificador Gaussiano
n_simulations = 500  # número de simulações de Monte Carlo
results = []

# 2. Implementação dos Modelos
def run_models(X_train, X_test, y_train, y_test):
    accuracies = {}

    # Modelo MQO tradicional - usaremos regressão linear
    model_mqo = GaussianNB()
    model_mqo.fit(X_train, y_train)
    accuracies["MQO tradicional"] = accuracy_score(y_test, model_mqo.predict(X_test))
    
    # Classificador Gaussiano Tradicional
    model_gaussian = QuadraticDiscriminantAnalysis(store_covariance=True)
    model_gaussian.fit(X_train, y_train)
    accuracies["Classificador Gaussiano Tradicional"] = accuracy_score(y_test, model_gaussian.predict(X_test))

    # Classificador Gaussiano (Covariância de todo o conjunto de treino)
    shared_cov_matrix = np.cov(X_train, rowvar=False)
    model_gaussian_cov_shared = QuadraticDiscriminantAnalysis(store_covariance=True, covariance_estimator=shared_cov_matrix)
    model_gaussian_cov_shared.fit(X_train, y_train)
    accuracies["Classificador Gaussiano (Cov. de todo cj. treino)"] = accuracy_score(y_test, model_gaussian_cov_shared.predict(X_test))

    # Classificador de Bayes Ingênuo
    model_nb = GaussianNB()
    model_nb.fit(X_train, y_train)
    accuracies["Classificador de Bayes Ingênuo"] = accuracy_score(y_test, model_nb.predict(X_test))

    # Classificadores Gaussianos Regularizados
    for lmbd in lambdas:
        shrink_cov = ShrunkCovariance(shrinkage=lmbd)
        model_reg = QuadraticDiscriminantAnalysis(store_covariance=True, covariance_estimator=shrink_cov)
        model_reg.fit(X_train, y_train)
        accuracies[f"Classificador Gaussiano Regularizado (λ={lmbd})"] = accuracy_score(y_test, model_reg.predict(X_test))

    return accuracies

# 3. Simulação de Monte Carlo
for _ in range(n_simulations):
    # Dividir em 80% treinamento e 20% teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Executar e armazenar as acurácias
    accuracies = run_models(X_train, X_test, y_train, y_test)
    results.append(accuracies)

# 4. Análise dos Resultados
# Calcular média, desvio padrão, valor máximo e mínimo para cada modelo
results_df = pd.DataFrame(results)
summary_df = pd.DataFrame({
    "Modelo": results_df.columns,
    "Média": results_df.mean(),
    "Desvio-Padrão": results_df.std(),
    "Maior Valor": results_df.max(),
    "Menor Valor": results_df.min()
}).reset_index(drop=True)

print(summary_df)

# 5. Visualização dos Resultados
summary_df.plot(x="Modelo", y="Média", yerr="Desvio-Padrão", kind="bar", capsize=5)
plt.ylabel("Acurácia Média")
plt.title("Desempenho dos Modelos de Classificação")
plt.xticks(rotation=45, ha="right")
plt.show()

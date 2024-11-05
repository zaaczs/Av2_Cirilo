import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Carregar os dados do arquivo
data = pd.read_csv(r"C:\Users\isaac\Downloads\aerogerador (1).dat", sep="\t", header=None)
data.columns = ["Wind_Speed", "Power_Output"]

# Definir o grau do polinômio (por exemplo, grau 3 para polinômio cúbico)
degree = 3

# Função para gerar a matriz de características polinomiais
def polynomial_features(X, degree):
    X_poly = np.ones((len(X), degree + 1))
    for i in range(1, degree + 1):
        X_poly[:, i] = (X.flatten() ** i)  # Achatar X para garantir compatibilidade
    return X_poly

# Preparar as variáveis de entrada e saída
X = data["Wind_Speed"].values.reshape(-1, 1)
y = data["Power_Output"].values

# Simulação para as 1000 rodadas de treinamento e teste
n_simulations = 1000
results = []

for _ in range(n_simulations):
    # Embaralhar e dividir os dados em 80% para treinamento e 20% para teste
    indices = np.random.permutation(len(X))
    train_size = int(0.8 * len(X))
    X_train, X_test = X[indices[:train_size]], X[indices[train_size:]]
    y_train, y_test = y[indices[:train_size]], y[indices[train_size:]]
    
    # Geração de características polinomiais
    X_train_poly = polynomial_features(X_train, degree)
    X_test_poly = polynomial_features(X_test, degree)
    
    # Calcular os coeficientes da regressão polinomial usando Mínimos Quadrados Ordinários
    theta_best = np.linalg.inv(X_train_poly.T.dot(X_train_poly)).dot(X_train_poly.T).dot(y_train)
    
    # Fazer previsões para o conjunto de teste
    y_pred = X_test_poly.dot(theta_best)
    rss = np.sum((y_test - y_pred) ** 2)  # Erro quadrático residual
    results.append(rss)

# Cálculo das métricas de desempenho
rss_mean = np.mean(results)
rss_std = np.std(results)
rss_max = np.max(results)
rss_min = np.min(results)

# Exibir os resultados
print("Resultados para a Regressão Polinomial (grau {}):".format(degree))
print("Média do RSS:", rss_mean)
print("Desvio-Padrão do RSS:", rss_std)
print("Maior RSS:", rss_max)
print("Menor RSS:", rss_min)

# Visualização dos dados de treino e da linha de regressão polinomial
X_poly = polynomial_features(X, degree)
y_pred_full = X_poly.dot(theta_best)

plt.scatter(data["Wind_Speed"], data["Power_Output"], color="blue", label="Dados Observados")
plt.plot(np.sort(X, axis=0), y_pred_full[np.argsort(X, axis=0)], color="red", label="Regressão Polinomial")
plt.xlabel("Velocidade do Vento")
plt.ylabel("Potência Gerada")
plt.title("Regressão Polinomial - Velocidade do Vento vs Potência Gerada")
plt.legend()
plt.show()

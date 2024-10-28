import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error

# Carregar os dados do arquivo
data = pd.read_csv("aerogerador.dat", sep="\t", header=None)  # ajuste os parâmetros se necessário
data.columns = ["Wind_Speed", "Power_Output"]

# 1. Visualização inicial dos dados
plt.scatter(data["Wind_Speed"], data["Power_Output"], color="blue")
plt.xlabel("Velocidade do Vento")
plt.ylabel("Potência Gerada")
plt.title("Gráfico de Dispersão: Velocidade do Vento vs Potência Gerada")
plt.show()

# 2. Organização dos dados em X e y
X = data[["Wind_Speed"]].values  # variável independente (velocidade do vento)
y = data["Power_Output"].values  # variável dependente (potência gerada)

# Parâmetros
lambdas = [0, 0.25, 0.5, 0.75, 1]  # valores de lambda para regularização
n_simulations = 500  # número de simulações de Monte Carlo
results = []

# 3. Implementação dos modelos
# Simulação Monte Carlo
for lmbd in lambdas:
    rss_values = []
    for _ in range(n_simulations):
        # Divisão dos dados em 80% treinamento e 20% teste
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        if lmbd == 0:
            # MQO tradicional
            model = LinearRegression()
        else:
            # MQO Regularizado (Tikhonov - Ridge Regression)
            model = Ridge(alpha=lmbd)
        
        # Treinar o modelo e fazer previsões
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calcular RSS e armazenar
        rss = mean_squared_error(y_test, y_pred) * len(y_test)
        rss_values.append(rss)
    
    # Armazenar as métricas para cada valor de lambda
    rss_mean = np.mean(rss_values)
    rss_std = np.std(rss_values)
    rss_max = np.max(rss_values)
    rss_min = np.min(rss_values)
    results.append((lmbd, rss_mean, rss_std, rss_max, rss_min))

# 4. Tabela de Resultados
results_df = pd.DataFrame(results, columns=["Lambda", "Média", "Desvio-Padrão", "Maior Valor", "Menor Valor"])
print(results_df)

# 5. Visualização dos Resultados
plt.errorbar(results_df["Lambda"], results_df["Média"], yerr=results_df["Desvio-Padrão"], fmt='-o', capsize=5)
plt.xlabel("Lambda")
plt.ylabel("Média do RSS")
plt.title("Média do RSS com Desvio-Padrão para cada valor de Lambda")
plt.show()

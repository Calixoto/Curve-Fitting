import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import norm


def base_radial(x, c, r):
    return np.exp(-(norm(x - c) / r) ** 2)


def ajuste_curvas(x, y, num_bases, r):
    # Criação das matrizes de design
    X = np.zeros((len(x), num_bases))
    for i in range(num_bases):
        for j in range(len(x)):
            X[j, i] = base_radial(x[j], x.mean() + i *
                                  (x.max() - x.min()) / (num_bases - 1), r)

    # Cálculo dos pesos
    w = np.linalg.lstsq(X, y, rcond=None)[0]

    # Predição dos valores ajustados
    y_pred = X.dot(w)

    return y_pred


# Carregar os dados do arquivo CSV
data = pd.read_csv('dados.csv', skipfooter=1, engine='python')
sum_data = pd.read_csv('dados.csv', nrows=1, skiprows=25,
                       usecols=range(1, 13), engine='python').columns

# Extrair os valores para o ano inteiro
x = np.arange(1, 13)
y = np.array(sum_data).astype(float)

# Remover NaNs e infs de x e y
print(x)
valid_indices = np.isfinite(x) & np.isfinite(y)
x = x[valid_indices]
y = y[valid_indices]

# Parâmetros do ajuste de curvas
num_bases = 100  # Número de funções de base radial
r = 1.0  # Parâmetro de escala das funções de base radial

# Ajuste de curvas
y_pred = ajuste_curvas(x, y, num_bases, r)

# Plotagem do resultado
plt.figure(figsize=(10, 6))
plt.plot(x, y, 'bo', label='Dados anuais')
plt.plot(x, y_pred, 'r-', label='Ajuste de curvas anual')
plt.legend()
plt.xlabel('Meses')
plt.ylabel('Intensidade')
plt.title('Ajuste de curvas anual')
plt.show()

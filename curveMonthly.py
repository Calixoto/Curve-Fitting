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
data = pd.read_csv('dados.csv')

data = pd.read_csv('dados.csv', skipfooter=1, engine='python')

# Extrair os valores para o mês de janeiro
x = np.array(data['Hora'])
y = np.array(data['Jan'])

# Remover NaNs e infs de x e y
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
plt.plot(x, y, 'bo', label='Dados mensais de janeiro')
plt.plot(x, y_pred, 'r-', label='Ajuste de curvas mensal de janeiro')
plt.legend()
plt.xlabel('Hora')
plt.ylabel('Intensidade')
plt.title('Ajuste de curvas mensal - Janeiro')
plt.show()

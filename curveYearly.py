import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Carregar os dados do arquivo CSV, ignorando a última linha
data = pd.read_csv('dados.csv', skipfooter=1, engine='python')

# Extrair os meses
meses = data.columns[1:]

# Extrair os dados de hora e valores mensais
hora = data['Hora'].values
valores_mensais = data[meses].values

# Função de ajuste - modelo polinomial de grau 2


def ajuste_anual(x, a, b, c):
    return a * x**2 + b * x + c


# Realizar o ajuste anual para cada ano
for i, ano in enumerate(valores_mensais):
    # Ajustar os dados anuais
    popt, pcov = curve_fit(ajuste_anual, hora, ano)

    # Gerar valores preditos para o ano atual
    y_pred = ajuste_anual(hora, *popt)

    # Plotar os dados reais e o ajuste anual
    plt.plot(hora, ano, 'o', label='Dados Reais')
    plt.plot(hora, y_pred, label='Ajuste Anual')

    plt.xlabel('Hora')
    plt.ylabel('Valores')
    plt.title(f'Ajuste Anual - Ano {meses[i]}')

    plt.legend()
    plt.show()

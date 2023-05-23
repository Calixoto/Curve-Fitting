import pandas as pd

datesFormatted = pd.read_csv("./report_CN.csv")

datesIntensity = datesFormatted.iloc[7:31, 1:13]
datesIntensity.columns = ['Hora', 'Jan', 'Fev', 'Mar', 'Abr',
                          'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov']

datesIntensity.to_csv('dados.csv', index=False)

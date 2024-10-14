import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

dataframe = pd.read_csv('datafile.csv')
dataframe['Nstencil'] = '(' + dataframe['N'].astype(str) + ', ' + dataframe['stencil'].astype(str) + ')'

plt.figure(figsize=(7, 14))
sns.boxplot(data=dataframe, x='Nstencil', y='time')
plt.title('Boxplot')
plt.xlabel('(N, Stencil)')
plt.ylabel('Time')
plt.show()

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

dataframe = pd.read_csv('datafile.csv')
dataframe['PNlead'] = '(' + dataframe['P'].astype(str) + ', ' + dataframe['N'].astype(str) + ', ' + dataframe['lead'].astype(str) +')'

plt.figure(figsize=(7, 14))
plt.xticks(rotation=45)
sns.boxplot(data=dataframe, x='PNlead', y='time')
plt.title('Boxplot')
plt.xlabel('(P, N, leader/noleader)')
plt.ylabel('Time')
plt.subplots_adjust(bottom=0.25, left=0.1, right=0.9, top=0.95)
plt.show()

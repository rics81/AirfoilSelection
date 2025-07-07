# -*- coding: utf-8 -*-
"""
Created on Sat Jul  5 12:53:30 2025

@author: Ricardo
"""
#%%
import pingouin as pg
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer import FactorAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

#%%
dfAirfoilFeat = dfAirfoilFeat.dropna()
dfAirfoilFeat = dfAirfoilFeat.reset_index(drop=True)

dfAirfoilFeat_names = dfAirfoilFeat['name']
dfAirfoilFeat = dfAirfoilFeat.drop('name', axis=1)

dfAirfoilFeatL = dfAirfoilFeat.loc[:,['r2max cl/cd alpha', 'r2clStdRate',
                                     'r2clStdStallRate', 'r2clMeanStallRate',
                                     'r2cdMax', 'r2cdStdRate', 'r2cdMeanRate', 'r2cmAlphaZero',
                                     'r5max cl/cd alpha', 'r5clStdRate',
                                     'r5clStdStallRate', 'r5clMeanStallRate',
                                     'r5cdMax', 'r5cdStdRate', 'r5cdMeanRate', 'r5cmAlphaZero',
                                     'r10max cl/cd alpha', 'r10clStdRate',
                                     'r10clStdStallRate', 'r10clMeanStallRate',
                                     'r10cdMax', 'r10cdStdRate', 'r10cdMeanRate', 'r10cmAlphaZero']]
dfAirfoilFeatG = dfAirfoilFeat.drop(dfAirfoilFeatL.columns, axis=1)
#%%
dfAirfoilFeat.info()
dfAirfoilFeatL.info()
dfAirfoilFeatG.info()

#%%
dfAirfoilFeat_desc = dfAirfoilFeat.describe()
dfAirfoilFeatL_desc = dfAirfoilFeatL.describe()
dfAirfoilFeatG_desc = dfAirfoilFeatG.describe()

# Initialize the StandardScaler
scaler = StandardScaler()

scaler.fit_transform(dfAirfoilFeatL)

# Create a box plot
dfAirfoilFeatLSTD = scaler.fit_transform(dfAirfoilFeatL)
plt.boxplot(scaler.fit_transform(dfAirfoilFeatLSTD))
plt.title('Basic Box Plot')
plt.ylabel('Value')
plt.show()

scaler.fit_transform(dfAirfoilFeatG)

# Create a box plot
dfAirfoilFeatGSTD = scaler.fit_transform(dfAirfoilFeatG)
plt.boxplot(scaler.fit_transform(dfAirfoilFeatGSTD))
plt.title('Basic Box Plot')
plt.ylabel('Value')
plt.show()
#%%
dfAirfoilFeat_rcorr = pg.rcorr(dfAirfoilFeat, method = 'pearson', upper = 'pval',
                               decimals = 4,
                               pval_stars = {0.01: '***', 0.05: '**', 0.10: '*'})

dfAirfoilFeatL_rcorr = pg.rcorr(dfAirfoilFeatL, method = 'pearson', upper = 'pval',
                               decimals = 4,
                               pval_stars = {0.01: '***', 0.05: '**', 0.10: '*'})

dfAirfoilFeatG_rcorr = pg.rcorr(dfAirfoilFeatG, method = 'pearson', upper = 'pval',
                               decimals = 4,
                               pval_stars = {0.01: '***', 0.05: '**', 0.10: '*'})

#%%
dfAirfoilFeat_corr = dfAirfoilFeat.corr()
dfAirfoilFeatL_corr = dfAirfoilFeatL.corr()
dfAirfoilFeatG_corr = dfAirfoilFeatG.corr()

#%%
bartlett, p_value = calculate_bartlett_sphericity(dfAirfoilFeat)
print(f'Qui² Bartlett: {round(bartlett, 2)}')
print(f'p-valor: {round(p_value, 4)}')

bartlett, p_value = calculate_bartlett_sphericity(dfAirfoilFeatL)
print(f'Qui² Bartlett: {round(bartlett, 2)}')
print(f'p-valor: {round(p_value, 4)}')

bartlett, p_value = calculate_bartlett_sphericity(dfAirfoilFeatG)
print(f'Qui² Bartlett: {round(bartlett, 2)}')
print(f'p-valor: {round(p_value, 4)}')

#%%
fa = FactorAnalyzer(n_factors=33, method='principal', rotation=None).fit(dfAirfoilFeat)
autovalores = fa.get_eigenvalues()[0]
print(autovalores) # Temos 4 autovalores, pois são 4 variáveis ao todo
round(autovalores.sum(), 2)

autovalores_fatores = fa.get_factor_variance()

tabela_eigen = pd.DataFrame(autovalores_fatores)
tabela_eigen.columns = [f"Fator {i+1}" for i, v in enumerate(tabela_eigen.columns)]
tabela_eigen.index = ['Autovalor','Variância', 'Variância Acumulada']
tabela_eigen = tabela_eigen.T

print(tabela_eigen)
# ---
faL = FactorAnalyzer(n_factors=24, method='principal', rotation=None).fit(dfAirfoilFeatL)
autovaloresL = faL.get_eigenvalues()[0]
print(autovaloresL) # Temos 4 autovalores, pois são 4 variáveis ao todo
round(autovaloresL.sum(), 2)

autovalores_fatoresL = faL.get_factor_variance()

tabela_eigenL = pd.DataFrame(autovalores_fatoresL)
tabela_eigenL.columns = [f"Fator {i+1}" for i, v in enumerate(tabela_eigenL.columns)]
tabela_eigenL.index = ['Autovalor','Variância', 'Variância Acumulada']
tabela_eigenL = tabela_eigenL.T

print(tabela_eigenL)
# ---
faG = FactorAnalyzer(n_factors=9, method='principal', rotation=None).fit(dfAirfoilFeatG)
autovaloresG = faG.get_eigenvalues()[0]
print(autovaloresG) # Temos 4 autovalores, pois são 4 variáveis ao todo
round(autovaloresG.sum(), 2)

autovalores_fatoresG = faG.get_factor_variance()

tabela_eigenG = pd.DataFrame(autovalores_fatoresG)
tabela_eigenG.columns = [f"Fator {i+1}" for i, v in enumerate(tabela_eigenG.columns)]
tabela_eigenG.index = ['Autovalor','Variância', 'Variância Acumulada']
tabela_eigenG = tabela_eigenG.T

print(tabela_eigenG)

#%%
fa = FactorAnalyzer(n_factors=10, method='principal', rotation=None).fit(dfAirfoilFeat)
autovalores = fa.get_eigenvalues()[0]
print(autovalores) # Temos 4 autovalores, pois são 4 variáveis ao todo
round(autovalores.sum(), 2)

autovalores_fatores = fa.get_factor_variance()

tabela_eigen = pd.DataFrame(autovalores_fatores)
tabela_eigen.columns = [f"Fator {i+1}" for i, v in enumerate(tabela_eigen.columns)]
tabela_eigen.index = ['Autovalor','Variância', 'Variância Acumulada']
tabela_eigen = tabela_eigen.T

print(tabela_eigen)
# ---
faL = FactorAnalyzer(n_factors=8, method='principal', rotation=None).fit(dfAirfoilFeatL)
autovaloresL = faL.get_eigenvalues()[0]
print(autovaloresL) # Temos 4 autovalores, pois são 4 variáveis ao todo
round(autovaloresL.sum(), 2)

autovalores_fatoresL = faL.get_factor_variance()

tabela_eigenL = pd.DataFrame(autovalores_fatoresL)
tabela_eigenL.columns = [f"Fator {i+1}" for i, v in enumerate(tabela_eigenL.columns)]
tabela_eigenL.index = ['Autovalor','Variância', 'Variância Acumulada']
tabela_eigenL = tabela_eigenL.T

print(tabela_eigenL)
# ---
faG = FactorAnalyzer(n_factors=3, method='principal', rotation=None).fit(dfAirfoilFeatG)
autovaloresG = faG.get_eigenvalues()[0]
print(autovaloresG) # Temos 4 autovalores, pois são 4 variáveis ao todo
round(autovaloresG.sum(), 2)

autovalores_fatoresG = faG.get_factor_variance()

tabela_eigenG = pd.DataFrame(autovalores_fatoresG)
tabela_eigenG.columns = [f"Fator {i+1}" for i, v in enumerate(tabela_eigenG.columns)]
tabela_eigenG.index = ['Autovalor','Variância', 'Variância Acumulada']
tabela_eigenG = tabela_eigenG.T

print(tabela_eigenG)

#%%
plt.figure(figsize=(24,16))
ax = sns.barplot(x=tabela_eigen.index, y=tabela_eigen['Variância'], data=tabela_eigen, palette='rocket')
ax.bar_label(ax.containers[0])
plt.title("Fatores Extraídos", fontsize=16)
plt.xlabel(f"{tabela_eigen.shape[0]} fatores que explicam {round(tabela_eigen['Variância'].sum()*100,2)}% da variância", fontsize=12)
plt.ylabel("Porcentagem de variância explicada", fontsize=12)
plt.show()

# ---
plt.figure(figsize=(24,16))
ax = sns.barplot(x=tabela_eigenL.index, y=tabela_eigenL['Variância'], data=tabela_eigenL, palette='rocket')
ax.bar_label(ax.containers[0])
plt.title("Fatores Extraídos", fontsize=16)
plt.xlabel(f"{tabela_eigenL.shape[0]} fatores que explicam {round(tabela_eigenL['Variância'].sum()*100,2)}% da variância", fontsize=12)
plt.ylabel("Porcentagem de variância explicada", fontsize=12)
plt.show()

# ---
plt.figure(figsize=(12,8))
ax = sns.barplot(x=tabela_eigenG.index, y=tabela_eigenG['Variância'], data=tabela_eigenG, palette='rocket')
ax.bar_label(ax.containers[0])
plt.title("Fatores Extraídos", fontsize=16)
plt.xlabel(f"{tabela_eigenG.shape[0]} fatores que explicam {round(tabela_eigenG['Variância'].sum()*100,2)}% da variância", fontsize=12)
plt.ylabel("Porcentagem de variância explicada", fontsize=12)
plt.show()
#%%
cargas_fatoriais = fa.loadings_

tabela_cargas = pd.DataFrame(cargas_fatoriais)
tabela_cargas.columns = [f"Fator {i+1}" for i, v in enumerate(tabela_cargas.columns)]
tabela_cargas.index = dfAirfoilFeat.columns

print(tabela_cargas)

# ---
cargas_fatoriaisL = faL.loadings_

tabela_cargasL = pd.DataFrame(cargas_fatoriaisL)
tabela_cargasL.columns = [f"Fator {i+1}" for i, v in enumerate(tabela_cargasL.columns)]
tabela_cargasL.index = dfAirfoilFeatL.columns

print(tabela_cargasL)

# ---
cargas_fatoriaisG = faG.loadings_

tabela_cargasG = pd.DataFrame(cargas_fatoriaisG)
tabela_cargasG.columns = [f"Fator {i+1}" for i, v in enumerate(tabela_cargasG.columns)]
tabela_cargasG.index = dfAirfoilFeatG.columns

print(tabela_cargasG)

#%%
plt.figure(figsize=(12,8))
tabela_cargas_chart = tabela_cargas.reset_index()
plt.scatter(tabela_cargas_chart['Fator 1'], tabela_cargas_chart['Fator 2'], s=50, color='red')

def label_point(x, y, val, ax):
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        ax.text(point['x'] + 0.05, point['y'], point['val'])

label_point(x = tabela_cargas_chart['Fator 1'],
            y = tabela_cargas_chart['Fator 2'],
            val = tabela_cargas_chart['index'],
            ax = plt.gca()) 

plt.axhline(y=0, color='grey', ls='--')
plt.axvline(x=0, color='grey', ls='--')
plt.ylim([-1.1,1.1])
plt.xlim([-1.1,1.1])
plt.title("Loading Plot", fontsize=16)
plt.xlabel(f"Fator 1: {round(tabela_eigen.iloc[0]['Variância']*100,2)}% de variância explicada", fontsize=12)
plt.ylabel(f"Fator 2: {round(tabela_eigen.iloc[1]['Variância']*100,2)}% de variância explicada", fontsize=12)
plt.show()

#%%
comunalidades = fa.get_communalities()

tabela_comunalidades = pd.DataFrame(comunalidades)
tabela_comunalidades.columns = ['Comunalidades']
tabela_comunalidades.index = dfAirfoilFeat.columns

print(tabela_comunalidades.sort_values(by='Comunalidades'))
# ---
comunalidades = faL.get_communalities()

tabela_comunalidadesL = pd.DataFrame(comunalidades)
tabela_comunalidadesL.columns = ['Comunalidades']
tabela_comunalidadesL.index = dfAirfoilFeatL.columns

print(tabela_comunalidadesL.sort_values(by='Comunalidades', ascending=False))
# ---
comunalidades = faG.get_communalities()

tabela_comunalidadesG = pd.DataFrame(comunalidades)
tabela_comunalidadesG.columns = ['Comunalidades']
tabela_comunalidadesG.index = dfAirfoilFeatG.columns

print(tabela_comunalidadesG.sort_values(by='Comunalidades', ascending=False))

#%%
fatores = pd.DataFrame(fa.transform(dfAirfoilFeat))
fatores.columns =  [f"Fator {i+1}" for i, v in enumerate(fatores.columns)]
dfAirfoilFeat_fatores = pd.concat([dfAirfoilFeat_names, fatores], axis=1)

# ---
fatores = pd.DataFrame(faL.transform(dfAirfoilFeatL))
fatores.columns =  [f"Fator {i+1}" for i, v in enumerate(fatores.columns)]
dfAirfoilFeatL_fatores = pd.concat([dfAirfoilFeat_names, fatores], axis=1)

# ---
fatores = pd.DataFrame(faG.transform(dfAirfoilFeatG))
fatores.columns =  [f"Fator {i+1}" for i, v in enumerate(fatores.columns)]
dfAirfoilFeatG_fatores = pd.concat([dfAirfoilFeat_names, fatores], axis=1)
#%%
scores = fa.weights_

tabela_scores = pd.DataFrame(scores)
tabela_scores.columns = [f"Fator {i+1}" for i, v in enumerate(tabela_scores.columns)]
tabela_scores.index = dfAirfoilFeat.columns

print(tabela_scores)

# ---
scores = faL.weights_

tabela_scoresL = pd.DataFrame(scores)
tabela_scoresL.columns = [f"Fator {i+1}" for i, v in enumerate(tabela_scoresL.columns)]
tabela_scoresL.index = dfAirfoilFeatL.columns

print(tabela_scoresL)

# ---
scores = faG.weights_

tabela_scoresG = pd.DataFrame(scores)
tabela_scoresG.columns = [f"Fator {i+1}" for i, v in enumerate(tabela_scoresG.columns)]
tabela_scoresG.index = dfAirfoilFeatG.columns

print(tabela_scoresG)
#%%
dfAirfoilFeat_fatores['Ranking'] = 0

for index, item in enumerate(list(tabela_eigen.index)):
    variancia = tabela_eigen.loc[item]['Variância']

    dfAirfoilFeat_fatores['Ranking'] = dfAirfoilFeat_fatores['Ranking'] + dfAirfoilFeat_fatores[tabela_eigen.index[index]]*variancia
    
print(dfAirfoilFeat_fatores)

#---
dfAirfoilFeatL_fatores['Ranking'] = 0

for index, item in enumerate(list(tabela_eigenL.index)):
    variancia = tabela_eigenL.loc[item]['Variância']

    dfAirfoilFeatL_fatores['Ranking'] = dfAirfoilFeatL_fatores['Ranking'] + dfAirfoilFeatL_fatores[tabela_eigenL.index[index]]*variancia
    
print(dfAirfoilFeatL_fatores)

#---
dfAirfoilFeatG_fatores['Ranking'] = 0

for index, item in enumerate(list(tabela_eigenG.index)):
    variancia = tabela_eigenG.loc[item]['Variância']

    dfAirfoilFeatG_fatores['Ranking'] = dfAirfoilFeatG_fatores['Ranking'] + dfAirfoilFeatG_fatores[tabela_eigenG.index[index]]*variancia
    
print(dfAirfoilFeatG_fatores)

#%%
dfAirfoilFeatL_fatores['Pos'] = dfAirfoilFeatL_fatores['Ranking'].rank(ascending=True)
dfAirfoilFeatG_fatores['Pos'] = dfAirfoilFeatG_fatores['Ranking'].rank(ascending=False)

dfAirfoilFeatRank = pd.merge(dfAirfoilFeatL_fatores['Pos'], dfAirfoilFeatG_fatores['Pos'], how='inner', left_index=True, right_index=True)
dfAirfoilFeatRank.columns = ['PosL', 'PosG']
dfAirfoilFeatRank['Final'] = dfAirfoilFeatRank['PosL'] + dfAirfoilFeatRank['PosG']

dfAirfoilFeatRank = pd.concat([dfAirfoilFeatRank, dfAirfoilFeat_names], axis=1)

# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 12:40:01 2025

@author: Ricardo
"""
from sklearn import preprocessing as skl_pp

# Inverte indicadores "quanto menor melhor"
inversoes = dfAirfoilFeatL.columns

dfAHP = dfAirfoilFeat.copy()

fnScaler=skl_pp.MinMaxScaler()
dfAHP=pd.DataFrame(fnScaler.fit_transform(dfAHP), columns=dfAHP.columns, index=dfAHP.index)

for col in dfAHP.columns:
    dfAHP[col][dfAHP[col]==0] = dfAHP[col][dfAHP[col]!=0].min()

for col in inversoes:
    dfAHP[col] = 1 / dfAHP[col]

# Normaliza
dfAHP = dfAHP.div(dfAHP.sum())

# Calcula pesos AHP
dfAHPStats = pd.DataFrame({
    'media': dfAHP.mean(),
    'desvio': dfAHP.std(),
    'cv': dfAHP.std() / dfAHP.mean()
})
dfAHPStats['peso'] = dfAHPStats['cv'] / dfAHPStats['cv'].sum()

# Aplica pesos
dfAHPRes = pd.DataFrame({'name': dfAirfoilFeat_names})
dfAHPRes['score_ahp'] = dfAHP.dot(dfAHPStats['peso'])
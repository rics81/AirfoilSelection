# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 10:39:01 2025

@author: Ricardo
"""

from sklearn import preprocessing as skl_pp

# Inverte indicadores "quanto menor melhor"
# dfAHP = dfAirfoilFeatL_fatores.iloc[:,1:-2].copy()
# dfAHP.columns = 'L_'+dfAHP.columns
# inversoes = dfAHP.columns
# dfAHP = pd.concat([dfAHP, dfAirfoilFeatG_fatores.iloc[:,1:-2]], axis=1)
# dfAHP = dfAHP.rename(columns={"Fator 1": "G_Fator 1", "Fator 2": "G_Fator 2", "Fator 3": "G_Fator 3"})

dfAHP = dfAirfoilFeat_fatores.iloc[:,1:-1].copy()
inversoes = dfAHP.columns[1:]

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
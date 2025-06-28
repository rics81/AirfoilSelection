# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 12:40:01 2025

@author: Ricardo
"""
dfAHP_G_data = pd.DataFrame(columns = ['name',
                                       'r2_clMax',
                                       'r2_clStdRate',
                                       'r2_clStdStallRate',
                                       'r2_clMeanStallRate',
                                       'r2_cdMax',
                                       'r2_cmAlphaZero',
                                       'r2_max cl/cd',
                                       'r5_clMax',
                                       'r5_clStdRate',
                                       'r5_clStdStallRate',
                                       'r5_clMeanStallRate',
                                       'r5_cdMax',
                                       'r5_cmAlphaZero',
                                       'r5_max cl/cd',
                                       'r10_clMax',
                                       'r10_clStdRate',
                                       'r10_clStdStallRate',
                                       'r10_clMeanStallRate',
                                       'r10_cdMax',
                                       'r10_cmAlphaZero',
                                       'r10_max cl/cd'])

df = {}
for airfoil in main_data:
    print(airfoil['name'])
   
    if airfoil['name'] not in failed_airfoils['name'].values:
        df.update({'name': airfoil['name']})
        
        for reynolds in ['200000', '500000', '1000000']:
            # if airfoil.get(reynolds) is None:
            #     failed_airfoils.append(airfoil['name'])
            #     print('There is no reynolds')
                
            # else:
            match reynolds:
                case '200000':
                    axColName = 'r2_'
                case '500000':
                    axColName = 'r5_'
                case _:
                    axColName = 'r10_'
                    
            # axColName = 'n5_' if reynolds == '200000-n5' else 'n9_'
            
            df.update({axColName + 'clMax': airfoil[reynolds]['clMax'],
                       axColName + 'clStdRate': airfoil[reynolds]['clStdRate'],
                       axColName + 'clStdStallRate': airfoil[reynolds]['clStdStallRate'],
                       axColName + 'clMeanStallRate': airfoil[reynolds]['clMeanStallRate'],
                       axColName + 'cdArea': airfoil[reynolds]['cdArea'],
                       axColName + 'cdMax': airfoil[reynolds]['cdMax'],
                       axColName + 'cmAlphaZero': airfoil[reynolds]['cmAlphaZero'],
                       axColName + 'max cl/cd': airfoil[reynolds]['max cl/cd']})
    
        dfAHP_G_data = pd.concat([dfAHP_G_data, pd.DataFrame(df, index=[0])])
    
dfAHP_G_data['r2_max cl/cd'] = pd.to_numeric(dfAHP_G_data['r2_max cl/cd'])
dfAHP_G_data['r5_max cl/cd'] = pd.to_numeric(dfAHP_G_data['r5_max cl/cd'])
dfAHP_G_data['r10_max cl/cd'] = pd.to_numeric(dfAHP_G_data['r10_max cl/cd'])

failed_airfoils.extend(list(dfAHP_G_data[dfAHP_G_data.isnull().any(axis=1)]['name']))
failed_airfoils = set(failed_airfoils)

airfoils_result = ahp(dfAHP_G_data[~dfAHP_G_data['name'].isin(failed_airfoils)])

# ----

"""Prepara dados para análise AHP."""
#df_norm = df.copy()
df_norm = dfAHP_G_data[~dfAHP_G_data['name'].isin(failed_airfoils)].copy()
df_norm = df_norm.reset_index(drop=True)

# Inverte indicadores "quanto menor melhor"
inversoes = ['r2_clStdRate', 'r2_clStdStallRate', 'r2_clMeanStallRate', 'r2_cdArea', 'r2_cdMax', 'r2_cmAlphaZero',
             'r5_clStdRate', 'r5_clStdStallRate', 'r5_clMeanStallRate', 'r5_cdArea', 'r5_cdMax', 'r5_cmAlphaZero',
             'r10_clStdRate', 'r10_clStdStallRate', 'r10_clMeanStallRate', 'r10_cdArea', 'r10_cdMax', 'r10_cmAlphaZero',]

adj_value = df_norm.loc[df_norm['r2_clStdStallRate'] != 0, 'r2_clStdStallRate'].min()
df_norm.loc[df_norm['r2_clStdStallRate'] == 0, 'r2_clStdStallRate'] = adj_value
adj_value = df_norm.loc[df_norm['r5_clStdStallRate'] != 0, 'r5_clStdStallRate'].min()
df_norm.loc[df_norm['r5_clStdStallRate'] == 0, 'r5_clStdStallRate'] = adj_value
adj_value = df_norm.loc[df_norm['r10_clStdStallRate'] != 0, 'r10_clStdStallRate'].min()
df_norm.loc[df_norm['r10_clStdStallRate'] == 0, 'r10_clStdStallRate'] = adj_value

df_norm.loc[df_norm['r2_max cl/cd alpha'] == 0, 'r2_max cl/cd alpha'] += 0.001
df_norm.loc[df_norm['r5_max cl/cd alpha'] == 0, 'r5_max cl/cd alpha'] += 0.001
df_norm.loc[df_norm['r10_max cl/cd alpha'] == 0, 'r10_max cl/cd alpha'] += 0.001

for col in inversoes:
    if col in df_norm.columns:
        df_norm[col] = 1 / df_norm[col]

# Normaliza
airfoils = df_norm['name']
df_norm = df_norm.drop(['name'], axis=1)
df_norm = df_norm.div(df_norm.sum())

# Calcula pesos AHP
df_stats = pd.DataFrame({
    'media': df_norm.mean(),
    'desvio': df_norm.std(),
    'cv': df_norm.std() / df_norm.mean()
})
df_stats['peso'] = df_stats['cv'] / df_stats['cv'].sum()

# Aplica pesos
df_resultado = pd.DataFrame({'name': airfoils})
df_resultado['score_ahp'] = df_norm.dot(df_stats['peso'])
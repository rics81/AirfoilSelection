# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 12:40:01 2025

@author: Iastech
"""

def ahp(df: pd.DataFrame) -> pd.DataFrame:
    """Prepara dados para análise AHP."""
    #df_norm = df.copy()
    df_norm = dfAHP_G_data[~dfAHP_G_data['name'].isin(failed_airfoils)].copy()
    df_norm = df_norm.reset_index(drop=True)
    
    # Inverte indicadores "quanto menor melhor"
    inversoes = ['r500_cdArea', 'r500_cdMeanRate', 'r500_cdStdRate', 'r500_cdMax', 'r500_clStdRate', 
                'r500_clStdStallRate', 'r500_cmMean', 'r500_cmStd', 'r500_max cl/cd alpha',
                'r1000_cdArea', 'r1000_cdMeanRate', 'r1000_cdStdRate', 'r1000_cdMax', 'r1000_clStdRate',
                'r1000_clStdStallRate', 'r1000_cmMean', 'r1000_cmStd', 'r1000_max cl/cd alpha']
    
    adj_value = df_norm.loc[df_norm['r500_clStdStallRate'] != 0, 'r500_clStdStallRate'].min()
    df_norm.loc[df_norm['r500_clStdStallRate'] == 0, 'r500_clStdStallRate'] = adj_value
    adj_value = df_norm.loc[df_norm['r1000_clStdStallRate'] != 0, 'r1000_clStdStallRate'].min()
    df_norm.loc[df_norm['r1000_clStdStallRate'] == 0, 'r1000_clStdStallRate'] = adj_value
   
    df_norm.loc[df_norm['r500_max cl/cd alpha'] == 0, 'r500_max cl/cd alpha'] += 0.001
    df_norm.loc[df_norm['r1000_max cl/cd alpha'] == 0, 'r1000_max cl/cd alpha'] += 0.001
    
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
    
    return df_resultado.sort_values('score_ahp', ascending=False)
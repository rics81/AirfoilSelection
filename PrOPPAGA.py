# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 16:48:34 2025

@author: Iastech
"""

dfPrOPPAGA = dfAirfoilFeat.apply(zscore, ddof=1).iloc[[37,297,298,296,293,160,180,475,208]]
w = {'weights': [33, 6, 30, 15, 18, 21, 24, 27, 9, 12, 3, 33, 6, 30, 15, 18, 21, 24, 27, 9, 12, 3, 33, 6, 30, 15, 18, 21, 24, 27, 9, 12, 3]}
dfPrOPPAGA_w = pd.DataFrame(data=w, index=dfPrOPPAGA.columns)
dfPrOPPAGA_w = dfPrOPPAGA_w/198

inversoes = dfAirfoilFeatL.columns

for col in inversoes:
    dfPrOPPAGA[col] = dfPrOPPAGA[col] * -1
    
dfPrOPPAGA = dfPrOPPAGA.apply(zscore, ddof=1)

dfPrOPPAGA['score'] = dfPrOPPAGA.dot(dfPrOPPAGA_w)

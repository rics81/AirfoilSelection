# -*- coding: utf-8 -*-
"""
Created on Sat Jul  5 11:44:06 2025

@author: Iastech
"""
import pandas as pd

# main_data[0]['re'][0].keys()
keys = ['max cl/cd', 'max cl/cd alpha', 'clMax', 'clStdRate', 'clMeanRate', 'clStdStallRate', 'clMeanStallRate', 'cdMax', 'cdStdRate', 'cdMeanRate', 'cmAlphaZero']

dfAirfoilFeat = []
for airfoil in main_data_filt:
    airfoildata = {'name': airfoil['name']}
    
    for re in airfoil['re']:
        match re['re']:
            case '200000':
                prefix = 'r2'
            
            case '500000':
                prefix = 'r5'
            
            case '1000000':
                prefix = 'r10'
        
        for key in keys:
            airfoildata[prefix + key]=re[key]
    
    #dfAirfoilFeat = pd.concat([dfAirfoilFeat, pd.DataFrame(airfoildata)])
    dfAirfoilFeat.append(airfoildata)

dfAirfoilFeat = pd.DataFrame(dfAirfoilFeat)

# dfAirfoilFeat['r2max cl/cd'] = pd.to_numeric(dfAirfoilFeat['r2max cl/cd'])
# dfAirfoilFeat['r5max cl/cd'] = pd.to_numeric(dfAirfoilFeat['r5max cl/cd'])
# dfAirfoilFeat['r10max cl/cd'] = pd.to_numeric(dfAirfoilFeat['r10max cl/cd'])
# dfAirfoilFeat['r2max cl/cd alpha'] = pd.to_numeric(dfAirfoilFeat['r2max cl/cd alpha'])
# dfAirfoilFeat['r5max cl/cd alpha'] = pd.to_numeric(dfAirfoilFeat['r5max cl/cd alpha'])
# dfAirfoilFeat['r10max cl/cd alpha'] = pd.to_numeric(dfAirfoilFeat['r10max cl/cd alpha'])
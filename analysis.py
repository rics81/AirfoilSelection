# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 16:28:21 2025

@author: Ricardo Sandrini
"""
import pandas as pd
import numpy as np
from scipy.integrate import simps

def check_airfoil_data(airfoil: dict) -> pd.DataFrame():
    failed_airfoil = pd.DataFrame(columns=['name', 'desc'])
    
    if len(airfoil['re']) < 3:        
        failed_airfoil = pd.DataFrame({'name':airfoil['name'], 'desc':['One or more Reynolds data is missing']})
        print(airfoil['name'] + " - One or more Reynolds data is missing")
    
    else: 
        for reynolds in airfoil['re']:
            al = reynolds['alpha']
            cl = reynolds['cl']
            cd = reynolds['cd']
            cm = reynolds['cm']
            
            al_ok = al.apply(pd.to_numeric, errors='coerce').notna().all()
            cl_ok = cl.apply(pd.to_numeric, errors='coerce').notna().all()
            cd_ok = cd.apply(pd.to_numeric, errors='coerce').notna().all()
            cm_ok = cm.apply(pd.to_numeric, errors='coerce').notna().all()
            
            if not al_ok or not cl_ok or not cd_ok or not cm_ok:
                failed_airfoil = pd.DataFrame({'name':airfoil['name'], 'desc':['alpha, cl, cd or cm series is corrupted']})
                print(airfoil['name'] + '- alpha, cl, cd or cm series is corrupted')
                break
            
            if len(cl) - np.argmax(cl) < 7:
                failed_airfoil = pd.DataFrame({'name':airfoil['name'], 'desc':['no stall area']})
                print(airfoil['name'] + '- no stall area')
                break
            
    return failed_airfoil

def add_data(re_data: dict):
    # Sample data (replace with your actual series)
    al = re_data['alpha']
    cl = re_data['cl']
    cd = re_data['cd']
    cm = re_data['cm']
    
    ## CL
    # --- Determine the max Cl--- ##
    re_data['clMax']=np.max(cl)
    
    # Find the index of the maximum value
    max_idx = np.argmax(cl)  # Index of max(y)
    
    # Filter y values that are positive and increasing until the maximum Cl value
    valid_indices = np.where((cl[:max_idx] > 0))[0]
    
    cl_valid = cl[valid_indices]
    
    # ---Determine the standard deviation of Cl increment ---##
    re_data['clStdRate']=np.diff(cl_valid).std()
    # ---Determine the mean of Cl increment --- ##
    re_data['clMeanRate']=np.diff(cl_valid).mean()
    
    # ---Determine the std value for decrease of Cl in stall area
    re_data['clStdStallRate']=np.diff(cl[max_idx:]).std()
    ## ---Determine the mean of Cl decrease in stall area --- ##
    re_data['clMeanStallRate']=abs(np.diff(cl[max_idx:]).mean())    
    
    ## CD
    valid_indices = np.where((al[:max_idx] >= 0))[0]
    
    cd_valid = cd[valid_indices]
    
    # airfoil[reynolds]['cdArea']=np.trapz(y_valid, x_valid)
    re_data['cdMax']=np.max(cd_valid)
    
    ## ---Determine the std value for increase of Cd
    re_data['cdStdRate']=np.diff(cd_valid).std()
    ## ---Determine the mean of Cd increase --- ##
    re_data['cdMeanRate']=np.diff(cd_valid).mean()
    
    ## CM    
    ## -- Determine value for cm when alpha is equal to zero
    re_data['cmAlphaZero'] = abs(cm[al >= 0].head(1).item())

# detecta "problemas" nos dados dos aerofolios
cnt = 0
samples = len(main_data)
failed_airfoils = pd.DataFrame(columns=['name', 'desc'])
for airfoil in main_data:
    failed_airfoils = pd.concat([failed_airfoils, check_airfoil_data(airfoil)])
    cnt += 1
    print (str(cnt) + '/' + str(samples))

# remove aerofolios com "problema" 
aux_data = []
for i, airfoil in enumerate(main_data):
    if airfoil['name'] not in list(failed_airfoils['name']):
        aux_data.append(airfoil)
main_data_filt = aux_data.copy()
del aux_data

# gera indicadores dos aerofolios
cnt = 0
samples = len(main_data_filt)
for airfoil in main_data_filt:
    print(airfoil['name'])
    
    for re in airfoil['re']:
        add_data(re)        
        
    cnt+=1
    print('total: '+str(cnt)+'/'+str(samples))

    # if airfoil['name'] == 'goe281-il':
    #   break
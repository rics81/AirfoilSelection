# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 16:28:21 2025

@author: Ricardo Sandrini
"""
import pandas as pd
import numpy as np
from scipy.integrate import simps

cnt=0
failed_airfoils=pd.DataFrame(columns=['name', 'desc'])
for airfoil in main_data:
    print(airfoil['name'])
    for reynolds in ['200000', '500000', '1000000']:
        
        if airfoil.get(reynolds) is None:
            failed_airfoils = pd.concat([failed_airfoils, pd.DataFrame({'name':airfoil['name'], 'desc':['There is no Reynolds']})])
            print(airfoil['name'] + '- There is no Reynolds')
            break
        
        # Sample data (replace with your actual series)
        x = airfoil[reynolds]['alpha']
        y = airfoil[reynolds]['cl']
        
        x_is_numeric = x.apply(pd.to_numeric, errors='coerce').notna().all()
        y_is_numeric = y.apply(pd.to_numeric, errors='coerce').notna().all()
        if not x_is_numeric or not y_is_numeric:
            if not airfoil['name'] in failed_airfoils['name']:
                failed_airfoils = pd.concat([failed_airfoils, pd.DataFrame({'name':airfoil['name'], 'desc':['cl, cd or alpha series is corrupted']})])
                print(airfoil['name'] + 'cl, cd or alpha series is corrupted')
            break
        
        ## ---
        ## --- Cl ANALYSIS DATA --- ##
        ## ---
        ## --- Determine the max Cl--- ##
        airfoil[reynolds]['clMax']=np.max(y)
        
        # Find the index of the maximum value
        max_idx = np.argmax(y)  # Index of max(y)
        
        # Filter y values that are positive and increasing until the maximum Cl value
        valid_indices = np.where((y[:max_idx] > 0))[0]
        
        x_valid = x[valid_indices]
        y_valid = y[valid_indices]
        
        # Compute the area using the trapezoidal rule
        #airfoil[reynolds]['clArea']=np.trapz(y_valid, x_valid)
        
        ## ---Determine the standard deviation of Cl increment ---##
        airfoil[reynolds]['clStdRate']=np.diff(y_valid[:max_idx]).std()
        ## ---Determine the mean of Cl increment --- ##
        # airfoil[reynolds]['clMeanRate']=np.diff(y[:max_idx]).mean()
        
        ##---Determine the std value for decrease of Cl in stall area
        airfoil[reynolds]['clStdStallRate']=np.diff(y[max_idx:]).std()
        ## ---Determine the mean of Cl decrease in stall area --- ##
        airfoil[reynolds]['clMeanStallRate']=abs(np.diff(y[max_idx:]).mean())
        
        ## ---
        ## --- Cl/Cd ANALYSIS DATA --- ##
        ## ---
        # Filter y values that have alpha greater than 0
        # y = airfoil[reynolds]['cl']/airfoil[reynolds]['cd']
        # y_valid = y[valid_indices]
        # airfoil[reynolds]['cl/cdRatioArea']=np.trapz(y_valid, x_valid)
        
        ## ---        
        ## --- Cd ANALYSIS DATA --- ##
        ## ---
        # Filter y values that have alpha greater than 0
        y = airfoil[reynolds]['cd']
        # valid_indices = np.where((x[:max_idx] >= 0))[0]
        # x_valid = x[valid_indices]
        y_valid = y[valid_indices]
        # airfoil[reynolds]['cdArea']=np.trapz(y_valid, x_valid)
        airfoil[reynolds]['cdMax']=np.max(y_valid)
        
        ## ---Determine the std value for increase of Cd
        # airfoil[reynolds]['cdStdRate']=np.diff(y[valid_indices]).std()
        ## ---Determine the mean of Cd increase --- ##
        # airfoil[reynolds]['cdMeanRate']=np.diff(y[valid_indices]).mean()
        
        ## ---        
        ## --- Cm ANALYSIS DATA --- ##
        ## ---
        y = airfoil[reynolds]['cm']
        ## ---Determine the std value for increase of Cd
        # airfoil[reynolds]['cmStd']=np.diff(y[valid_indices]).std()
        ## ---Determine the mean of Cd increase --- ##
        # airfoil[reynolds]['cmMean']=np.diff(y[valid_indices]).mean()
        ## -- Determine value for cm when alpha is equal to zero
        airfoil[reynolds]['cmAlphaZero'] = abs(y[x >= 0].head(1).item())
        
        #break
    
    cnt+=1
    print('total: '+str(cnt)+'/'+str(len(main_data)))

    # if airfoil['name'] == 'goe281-il':
    #     break
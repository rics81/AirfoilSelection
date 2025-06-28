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
    for reynolds in ['1000000-n5', '1000000']:
        
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
        airfoil[reynolds]['clStdRate']=np.diff(y[:max_idx]).std()
        ## ---Determine the mean of Cl increment --- ##
        # airfoil[reynolds]['clMeanRate']=np.diff(y[:max_idx]).mean()
        
        ##---Determine the std value for decrease of Cl in stall area
        airfoil[reynolds]['clStdStallRate']=np.diff(y[max_idx:]).std()
        ## ---Determine the mean of Cl decrease in stall area --- ##
        airfoil[reynolds]['clMeanStallRate']=np.diff(y[max_idx:]).mean()
        
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
        valid_indices = np.where((x[:max_idx] > 0))[0]
        x_valid = x[valid_indices]
        y_valid = y[valid_indices]
        airfoil[reynolds]['cdArea']=np.trapz(y_valid, x_valid)
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
        airfoil[reynolds]['cmAlphaZero'] = y[x >= 0].head(1).item()
        
        #break
    
    cnt+=1
    print('total: '+str(cnt)+'/'+str(len(main_data)))

    # if airfoil['name'] == 'goe281-il':
    #     break

dfAHP_G_data = pd.DataFrame(columns = ['name',
                                       'n5_clMax',
                                       'n5_clStdRate',
                                       'n5_clStdStallRate',
                                       'n5_clMeanStallRate',
                                       'n5_cdArea',
                                       'n5_cdMax',
                                       'n5_cmAlphaZero',
                                       'n5_max cl/cd',
                                       'n9_clMax',
                                       'n9_clStdRate',
                                       'n9_clStdStallRate',
                                       'n9_clMeanStallRate',
                                       'n9_cdArea',
                                       'n9_cdMax',
                                       'n9_cmAlphaZero',
                                       'n9_max cl/cd'])

df = {}
for airfoil in main_data:
    print(airfoil['name'])
   
    if airfoil['name'] not in failed_airfoils['name'].values:
        df.update({'name': airfoil['name']})
        
        for reynolds in ['1000000-n5', '1000000']:
            # if airfoil.get(reynolds) is None:
            #     failed_airfoils.append(airfoil['name'])
            #     print('There is no reynolds')
                
            # else:            
            axColName = 'n5_' if reynolds == '1000000-n5' else 'n9_'
            
            df.update({axColName + 'clMax': airfoil[reynolds]['clMax'],
                       axColName + 'clStdRate': airfoil[reynolds]['clStdRate'],
                       axColName + 'clStdStallRate': airfoil[reynolds]['clStdStallRate'],
                       axColName + 'clMeanStallRate': airfoil[reynolds]['clMeanStallRate'],
                       axColName + 'cdArea': airfoil[reynolds]['cdArea'],
                       axColName + 'cdMax': airfoil[reynolds]['cdMax'],
                       axColName + 'cmAlphaZero': airfoil[reynolds]['cmAlphaZero'],
                       axColName + 'max cl/cd': airfoil[reynolds]['max cl/cd']})
    
        dfAHP_G_data = pd.concat([dfAHP_G_data, pd.DataFrame(df, index=[0])])
    
dfAHP_G_data['n5_max cl/cd'] = pd.to_numeric(dfAHP_G_data['n5_max cl/cd'])
dfAHP_G_data['n9_max cl/cd'] = pd.to_numeric(dfAHP_G_data['n9_max cl/cd'])

failed_airfoils.extend(list(dfAHP_G_data[dfAHP_G_data.isnull().any(axis=1)]['name']))
failed_airfoils = set(failed_airfoils)

airfoils_result = ahp(dfAHP_G_data[~dfAHP_G_data['name'].isin(failed_airfoils)])

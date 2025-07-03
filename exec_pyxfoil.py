# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 17:37:52 2025

@author: Ricardo
"""
from pyxfoil import Xfoil, set_workdir, set_xfoilexe
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
import tempfile
import os, shutil

def delete_aux_files(path: str):
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def build_airfoil_string(data_dict) -> StringIO:
    name = data_dict['name']
    df = data_dict['coord']

    output = StringIO()
    output.write(f"{name}\n")
    for _, row in df.iterrows():
        output.write(f"{row['x']:.5f}     {row['y']:.5f}\n")
    
    output.seek(0)  # Move to the start if you need to read from it later
    return output

def get_polar_data(airfoil: dict,
                   folder: str,
                   alpha_st: float,
                   alpha_end: float,
                   alpha_step: float,
                   mach: float,
                   re: float):
    
    delete_aux_files(folder)
    
    file = build_airfoil_string(airfoil)
    
    xfoil = Xfoil('test')
    
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.dat', delete=False, dir = xfoil_folder + 'files') as temp_file:
        temp_file.write(file.getvalue())
        temp_file.flush()
        # Now use that temporary file with pyxfoil
        xfoil.points_from_dat(temp_file.name)
        xfoil.name = 'test'

    xfoil.set_ppar(180)
    
    polar_data = xfoil.run_polar(alpha_st, alpha_end, alpha_step, mach=mach, re=re)
    return polar_data
    
# Import Dependencies
# https://github.com/Xero64/pyxfoil?tab=readme-ov-file
xfoil_folder = 'C:\\temp\\xfoil\\'

set_workdir(xfoil_folder + 'files') # Sets the working directory for pyxfoil.
set_xfoilexe(xfoil_folder + 'xfoil.exe') # Sets the path of the xfoil executable.

cnt = 0
total = len(airfoil_coords)
airfoil_polars = []

for airfoil in airfoil_coords:
    print ('airfoil: ' + airfoil['name'])
    print (str(cnt) + '/' + str(total))
    
    polars = get_polar_data(airfoil, xfoil_folder + 'files', -5.0, 20.0, 0.25, 0.1, 500000)

    if len(polars.alpha)>=80:
        polar_data = {'name': airfoil['name']}
        polar_data['polars'] = polars
        airfoil_polars.append(polar_data)

    cnt += 1
    if cnt>=10: break
    break

# Runs xfoil for the following parameters
al = pd.Series(np.arange(-5.0, 20.0, 0.5))

for ali in al:
    rescase = xfoil.run_result(ali, mach=mach, re=Re)
   
xfoil2.points_from_dat('C://Users//Iastech//Documents//Repo//Airfoiltools//airfoils//n9.dat')

try:
    polar=
except OSError as err:
    print("OS error:", err)
    
xfoil2.run_polar(-5.0, 15.0, 0.5, mach=mach, re=Re)

# Plot two polars created above
axp1 = None
axp1 = polar1.plot_polar(ax=axp1)
_ = axp1.legend()

plt.figure()
plt.plot(polar1.alpha, polar1.cl)
plt.show()
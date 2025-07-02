# -*- coding: utf-8 -*-
"""
Created on Sun Jun 29 14:25:13 2025

@author: Ricardo
"""
# Libraries
import requests
import pandas as pd
from bs4 import BeautifulSoup
from io import StringIO

def ufn_get_dat_links() -> list[str]:
    # https://m-selig.ae.illinois.edu/ads/coord_database.html
    hdr = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
        'Accept-Language': 'pt-BR,pt;q=0.9',
        'DNT': '1'
    }
    url = "https://m-selig.ae.illinois.edu/ads/coord_database.html"
    
    response = requests.get(url, headers=hdr, timeout=10)
    response.raise_for_status()
    
    soup = BeautifulSoup(response.text, 'html.parser')            
    dat_links = [a['href'] for a in soup.find_all('a', href=True) if a['href'].endswith('.dat')]        
    return dat_links

def ufn_get_dat_files(dat_url: str) -> StringIO:
    hdr = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
        'Accept-Language': 'pt-BR,pt;q=0.9',
        'DNT': '1'
    }
    
    response = requests.get(dat_url, headers=hdr, timeout=10)
    response.raise_for_status()
    dat_file = StringIO(response.text)
    return dat_file

def format_coord_data(url: str) -> dict:
    try:
        file = ufn_get_dat_files(url)
    except requests.exceptions.HTTPError as e:
        failed_url = {'url':url}
        return failed_url
        print("ufn_get_dat_files - HTTP error occurred:", e)
    except requests.exceptions.RequestException as e:
        failed_url = {'url': url}
        return failed_url
        print("ufn_get_dat_files - A request error occurred:", e)

    #le arquivo e qubra linhas
    lines = file.getvalue().splitlines()
    
    while lines[-1].strip() == '':
        lines.pop()
        
    # retira espaços extras do começo e do fim da string e detecta linhas em branco
    file_str = [len(item.strip()) for item in lines]
    # guarda linhas que estão em branco. por ex.: [2, 70]
    file_str = [i for i, lenght in enumerate(file_str) if lenght==0]
    
    data = {'url': url}
    data['name'] = lines[0]

    if len(file_str) == 2:
        # guarda dados do extradorso com x decrescente
        coords = [coord.strip().split() for coord in lines[file_str[0]+1:file_str[1]]]
        data['coord'] = pd.DataFrame(coords, columns=['x', 'y']).sort_values(by='x', ascending=False)
        # guarda dados do intradorso com x crescente
        coords = pd.DataFrame([coord.strip().split() for coord in lines[file_str[1]:]], columns=['x', 'y']).sort_values(by='x')
        data['coord'] = pd.concat([data['coord'], coords])
        
    else:
        data['coord'] = pd.DataFrame([coord.strip().split() for coord in lines[1:]], columns=['x', 'y'])
    
    # remove dados não numericos
    data['coord'] = data['coord'][pd.to_numeric(data['coord']['x'], errors='coerce').notna()]
    data['coord']['x'] = pd.to_numeric(data['coord']['x'], errors='coerce')
    data['coord']['y'] = pd.to_numeric(data['coord']['y'], errors='coerce')
    # ajusta indíce
    data['coord'] = data['coord'].reset_index(drop=True)
    
    # ajusta ultimo registro se x não termina na origem
    if data['coord']['x'][0] != data['coord']['x'].iloc[-1]:
        data['coord'] = pd.concat([data['coord'],
                                   pd.DataFrame({'x':data['coord']['x'][0],
                                                 'y':data['coord']['y'][0]},
                                                index=[len(data['coord'])])
                                   ])
    return data

# --MAIN
try:
    uiuc_dat_links = ufn_get_dat_links()
    uiuc_dat_links = ['https://m-selig.ae.illinois.edu/ads/' + item for item in uiuc_dat_links]
except requests.exceptions.HTTPError as e:
    print("ufn_get_dat_links - HTTP error occurred:", e)
except requests.exceptions.RequestException as e:
    print("ufn_get_dat_links - A request error occurred:", e)

cnt = 0
total = len(uiuc_dat_links)
 
airfoil_coords=[]
airfoil_coords_failed=[]
for url in uiuc_dat_links:
    coords_data = format_coord_data(url)
    if len(list(coords_data.keys()))>1:
        airfoil_coords.append(coords_data)
        cnt+= 1
        print(f'airfoil {cnt}/{total} - {url}')
    else:
        airfoil_coords_failed.append(coords_data['url'])
        print(f'failed airfoil - {url}')
    
    #if cnt>=10: break
    
# https://m-selig.ae.illinois.edu/ads/coord/fx74130wp2mod.dat
# -*- coding: utf-8 -*-
"""
Created on Thu May 29 10:26:38 2025

@author: Ricardo Sandrini
"""
# <codecell>
# Bibliotecas
import requests
import pandas as pd
from bs4 import BeautifulSoup
from io import StringIO
import re


# <codecell>
# Configurações de scraping
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
    'Accept-Language': 'pt-BR,pt;q=0.9',
    'DNT': '1'
}

# <codecell>
def delete_first_n_lines(text, n):
    lines = text.splitlines(True)
    return "".join(lines[n:])

# <codecell>
url = "http://airfoiltools.com/search/airfoils"
response = requests.get(url, headers=HEADERS, timeout=10)
response.raise_for_status()

soup = BeautifulSoup(response.text, 'html.parser')

#print(soup.prettify())

tables = soup.find_all('table')

# for i, table in enumerate(tables):
#     with open(f'html_{i}.txt', 'w') as file:
#         t = f"Table {i}:\n" + table.prettify() + "\n" + "-"*50
#         file.write(t)
#         #print(f"Table {i}:\n", table.prettify(), "\n" + "-"*50)
        
# Find all links within the table
links = tables[1].find_all('a')

# Extract name and href attributes
data = []
for link in links:
    name = link.get_text(strip=True)
    href = link.get('href')
    if name and href:
        data.append({'Name': name, 'Link_Detail': href})

# Convert to DataFrame
dfURLs = pd.DataFrame(data)

dfURLs['Link_Polar'] = dfURLs['Link_Detail'].str.replace("/airfoil/details?airfoil=", "http://airfoiltools.com/polar/csv?polar=xf-")
dfURLs['Link_Detail'] = 'http://airfoiltools.com' + dfURLs['Link_Detail']

cnt = 0
samples = len(dfURLs)
main_data = []
for index, airfoil in dfURLs.iterrows():
    
    ax_dict={'name': airfoil['Name'].split(' - ')[0],
             'url': airfoil['Link_Detail'],
             're': []}
    
    for reynolds in ['200000', '500000', '1000000']:
        url = airfoil['Link_Polar'] + '-' + reynolds
        response = requests.get(url)
        if response.status_code == 200:
            
            csv_file = StringIO(response.text)            
            df_header = pd.read_csv(csv_file, nrows=9, sep=',', on_bad_lines='skip')
            
            csv_file = StringIO(delete_first_n_lines(response.text, 10))
            df_data = pd.read_csv(csv_file, on_bad_lines='skip')            
            
            ax_dict['desc'] = airfoil['Name'].split(' - ')[1]
            ax_dict['n_crit'] = df_header.loc['Ncrit'].item()
            
            ax_dict['re'].append({
                're': reynolds,
                'max cl/cd': df_header.loc['Max Cl/Cd'].item(),
                'max cl/cd alpha': df_header.loc['Max Cl/Cd alpha'].item(),
                'alpha': df_data['Alpha'],
                'cl': df_data['Cl'],
                'cd': df_data['Cd'],
                'cdp': df_data['Cdp'],
                'cm': df_data['Cm'],
                'top_xtr': df_data['Top_Xtr'],
                'bot_xtr': df_data['Bot_Xtr']
            })
            
            print(response.content)
            print("CSV file downloaded successfully!")
        else:
            print("Failed to download file, status code:", response.status_code)
        
        #break
    
    response = requests.get(airfoil['Link_Detail'], headers=HEADERS, timeout=10)
    response.raise_for_status()
    
    soup = BeautifulSoup(response.text, 'html.parser')
    tables = soup.find_all("td", {"class": "cell1"})    

    # Extract all text parts
    text_parts = soup.get_text(separator="\n", strip=True).split("\n")
    
    # Extract specific elements
    name = text_parts[0]  # First line contains the name
    #desc = text_parts[1]  # Second line contains the description
    
    # Use regex to extract thickness and camber values
    thickness_match = re.search(r"Max thickness ([\d\.]+)% at ([\d\.]+)% chord", " ".join(text_parts))
    camber_match = re.search(r"Max camber ([\d\.]+)% at ([\d\.]+)% chord", " ".join(text_parts))
    
    ax_dict['thickness'] = {"value": float(thickness_match.group(1)), "chord": float(thickness_match.group(2))} if thickness_match else None
    ax_dict['camber'] = {"value": float(camber_match.group(1)), "chord": float(camber_match.group(2))} if camber_match else None
    
    main_data.append(ax_dict)
    
    cnt+=1
    print(url+'\n'+'total: '+str(cnt)+'/'+str(samples))

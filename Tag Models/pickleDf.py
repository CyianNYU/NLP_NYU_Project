import os
import numpy as np
import pandas as pd
import pickle as pkl

#Set for your computer
data_directory = '/'.join(os.getcwd().split("/")[:-2]) + '/data/'

def loadCleanedDf():

    orig_data = pd.read_csv(data_directory + 'qaData.csv', parse_dates=['Date'])
    orig_data['EarningTag2'] = orig_data['EarningTag2'].str.strip()

    #Add Year and Month, Quarter from Data
    orig_data['Year'] = orig_data['Date'].dt.year
    orig_data['Month'] = orig_data['Date'].dt.month
    orig_data['Quarter'] = orig_data['Month'].apply(lambda x: 1 if x < 4 else 2 if x < 7 else 3 if x < 9 else 4)
    orig_data['Company'] = orig_data['Company'].str.title().str.replace(" ", "")
    orig_data['EventType'] = orig_data['EventType'].str.title().str.replace(" ", "")
    orig_data['AnalystName'] = orig_data['AnalystName'].str.title().str.replace(" ", "")
    orig_data['Tag'] = orig_data['EarningTag2'].str.title().str.replace(" ", "")

    orig_data = orig_data.loc[~orig_data['AnalystName'].isna()].copy()

    #Index Data
    groups = []
    for i, (name, group) in enumerate(orig_data.groupby(['Company', 'Month', 'Year', 'Quarter', 'EventType', 'Date'])):
        g2 = group.copy()
        g2['EventNumber'] = i
        g2.reset_index(drop=True, inplace=True)
        g2.index.name = "QuestionNumber"
        g2.reset_index(inplace=True)
        groups.append(g2)

    indexed_data = pd.concat(groups)[['EventNumber', 'QuestionNumber', 'Company', 'Month', 'Year', 'Quarter', 'EventType', 'Date', 'AnalystName', "Tag"]]
    
    with open(data_directory+"cleaned_data.p", "wb") as f:
        pkl.dump(indexed_data, f)
    return "success"

print(loadCleanedDf())
    

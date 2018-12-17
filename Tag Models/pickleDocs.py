import os

import pickle as pkl
import re

import scipy as sp
import numpy as np
import pandas as pd

import nimfa 

from sklearn.preprocessing import normalize

from gensim.models.phrases import Phrases, Phraser

data_directory = '/'.join(os.getcwd().split("/")[:-2]) + '/data/'

import sys
sys.path.insert(3, '/'.join(os.getcwd().split("/")[:-2]) + '/textacy')

import textacy
from textacy import preprocess_text, Doc, Corpus
from textacy.vsm import Vectorizer, GroupVectorizer
from textacy.tm import TopicModel
en = textacy.load_spacy("en_core_web_sm", disable='parser')


test_set = [173,  74,  20, 101,  83,   1,  38,  39,  72,  50,  21, 164,  57,
       169, 8,  63, 102,  34,  80, 192, 139,  88, 112, 116,  61,  46,
        51, 165, 135,  89, 108,   7,  25,  15, 125,  93, 130,  71]
        
def loadTokenizedDocs():
    orig_data = pd.read_csv(data_directory + 'qaData.csv', parse_dates=['Date'])
    orig_data['Year'] = orig_data['Date'].dt.year
    orig_data['Month'] = orig_data['Date'].dt.month
    orig_data['Quarter'] = orig_data['Month'].apply(lambda x: 1 if x < 4 else 2 if x < 7 else 3 if x < 9 else 4)
    orig_data['Company'] = orig_data['Company'].str.title().str.replace(" ", "")
    orig_data['AnalystName'] = orig_data['AnalystName'].str.title().str.replace(" ", "")
    orig_data['Tag'] = orig_data['EarningTag2'].str.title().str.replace(" ", "")

    orig_data = orig_data.loc[~orig_data['AnalystName'].isna()].copy()

    groups = []
    for i, (name, group) in enumerate(orig_data.groupby(['Company', 'Month', 'Year', 'Quarter', 'EventType', 'Date'])):
        g2 = group.copy()
        g2['EventNumber'] = i
        g2.reset_index(drop=True, inplace=True)
        g2.index.name = "QuestionNumber"
        g2.reset_index(inplace=True)
        groups.append(g2)

    indexed_data = pd.concat(groups)
    q_data = indexed_data[['Date', 'EventNumber', 'QuestionNumber', 'Year', 'Quarter', 'Company', 'AnalystName', 'EventType', 'Tag', 'Question']].copy()


    docs = Corpus(lang=en, docs=q_data.apply(lambda x: Doc(content=' '.join(
                                                            [token for token in preprocess_text(text=x['Question'], lowercase=True, no_punct=True, no_contractions=True, no_accents=True, no_currency_symbols=True, no_numbers=True).split(' ') if len(token)>2]),
                                                        lang=en, metadata={'Year':x['Year'],
                                                                           'Quarter':x['Quarter'],
                                                                           'Company':x['Company'],
                                                                           'AnalystName':x["AnalystName"],
                                                                           'Tag':x['Tag'],
                                                                           'EventType':x['EventType'],
                                                                           'EventNumber':x['EventNumber'],
                                                                           'QuestionNumber':x['QuestionNumber']}),axis=1).tolist())

    tokenized_docs = [[list(doc.to_terms_list(ngrams=(1), as_strings=True, normalize='lemma', drop_determiners=True)), doc.metadata] for doc in docs]
    
    with open(data_directory+"tokenized_docs.p", "wb") as f:
        pkl.dump(tokenized_docs, f)
    return "success"
        
print(loadTokenizedDocs())
    

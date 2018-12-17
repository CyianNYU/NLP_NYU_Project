import os

import pickle as pkl
import re

import scipy as sp
import numpy as np
import pandas as pd

import nimfa 

from sklearn.preprocessing import normalize
from sklearn.decomposition import NMF

#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams["xtick.labelsize"] = 7
#plt.style.use('dark_background')
import seaborn as sns
#sns.set(context='poster', style='dark', rc={'figure.facecolor':'white'}, font_scale=1.2)

from gensim.models.phrases import Phrases, Phraser

data_directory = '/'.join(os.getcwd().split("/")[:-2]) + '/data/'
figure_directory = '/'.join(os.getcwd().split("/")[:-1]) + '/figures/'

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
        
        

orig_data = pd.read_csv(data_directory + 'qaData.csv', parse_dates=['Date'])
orig_data['Year'] = orig_data['Date'].dt.year
orig_data['Month'] = orig_data['Date'].dt.month
orig_data['Quarter'] = orig_data['Month'].apply(lambda x: 1 if x < 4 else 2 if x < 7 else 3 if x < 9 else 4)
orig_data['Company'] = orig_data['Company'].str.title().str.replace(" ", "")
orig_data['AnalystName'] = orig_data['AnalystName'].str.title().str.replace(" ", "")
orig_data['Tag'] = orig_data['EarningTag2'].str.title().str.replace(" ", "")

orig_data = orig_data.loc[~orig_data['AnalystName'].isna()].copy()

groups = []
for i, (name, group) in enumerate(orig_data.groupby(['Company', 'Participants', 'Month', 'Year', 'Quarter', 'EventType', 'Date'])):
    g2 = group.copy()
    g2['EventNumber'] = i
    groups.append(g2)
    
indexed_data = pd.concat(groups)
#train_data = indexed_data.loc[~indexed_data['EventNumber'].isin(test_set)]
q_data = indexed_data[['Date', 'EventNumber', 'Year', 'Quarter', 'Company', 'AnalystName', 'EventType', 'Tag', 'Question']].copy()


docs = Corpus(lang=en, docs=q_data.apply(lambda x: Doc(content=' '.join(
                                                        [token for token in preprocess_text(text=x['Question'], lowercase=True, no_punct=True, no_contractions=True, no_accents=True, no_currency_symbols=True, no_numbers=True).split(' ') if len(token)>2]),
                                                    lang=en, metadata={'Year':x['Year'],
                                                                       'Quarter':x['Quarter'],
                                                                       'Company':x['Company'],
                                                                       'AnalystName':x["AnalystName"],
                                                                       'Tag':x['Tag'],
                                                                       'EventType':x['EventType'],
                                                                       'EventNumber':x['EventNumber']}),axis=1).tolist())

tokenized_docs = [[list(doc.to_terms_list(ngrams=(1), as_strings=True, normalize='lemma', drop_determiners=True)), doc.metadata] for doc in docs if doc.metadata['EventNumber'] not in test_set]

bigram_phraser = Phraser(Phrases([doc[0] for doc in tokenized_docs], min_count=10, threshold=20, delimiter=b' '))
bigram_docs = [bigram_phraser[doc[0]] for doc in tokenized_docs] 

trigram_phraser = Phraser(Phrases(bigram_docs, min_count=5, threshold=10, delimiter=b' '))
trigram_docs = [trigram_phraser[doc] for doc in bigram_docs] 

analysts = [d[1]['AnalystName'] for d in tokenized_docs]
tags = [d[1]['Tag'] for d in tokenized_docs]
companies = [d[1]['Company'] for d in tokenized_docs]

print("Starting Modeling")

###############################################################################
##Analyst BM25
###############################################################################

analysts_vec = GroupVectorizer(tf_type='bm25', apply_idf=True, idf_type='smooth', apply_dl=True, dl_type='linear').fit(trigram_docs, analysts)
analyst_doc_term_matrix = analysts_vec.transform(trigram_docs, analysts)

mod = nimfa.Nmf(V=analyst_doc_term_matrix, max_iter=200, rank=5)
nmf_grid = nimfa.Nmf.estimate_rank(mod, rank_range=np.arange(2, 25), what=['cophenetic', 'rss'], n_run=20)


fig = plt.figure(figsize=(8,6))
fig_plt = sns.barplot(x=np.arange(2,25), y=[i['cophenetic'] for i in nmf_grid.values()])
fig_plt.set_xlabel("N-Components")
fig_plt.set_ylabel("Cophenetic")
fig_plt.set_title('Cophenetic Score vs. N-Components')
print(fig_plt)
fig_plt.get_figure().savefig(figure_directory + "NMF/BM25/AnalystCophenticScores.png", bbox_inches='tight')


fig = plt.figure(figsize=(8,6))
fig_plt = sns.barplot(x=np.arange(2,25), y=[i['rss'] for i in nmf_grid.values()])
fig_plt.set_xlabel("N-Components")
fig_plt.set_ylabel("RSS")
fig_plt.set_title('RSS vs. N-Components')
print(fig_plt)
fig_plt.get_figure().savefig(figure_directory + "NMF/BM25/AnalystRSSScores.png", bbox_inches='tight')

print("Finished Analyst BM25")        

###############################################################################
##Tag BM25
###############################################################################

tag_vec = GroupVectorizer(tf_type='bm25', apply_idf=True, idf_type='smooth', apply_dl=True, dl_type='linear').fit(trigram_docs, tags)
tag_doc_term_matrix = tag_vec.transform(trigram_docs, tags)

mod = nimfa.Nmf(V=tag_doc_term_matrix, max_iter=200, rank=5)
nmf_grid = nimfa.Nmf.estimate_rank(mod, rank_range=np.arange(2, 15), what=['cophenetic', 'rss'], n_run=10)


fig = plt.figure(figsize=(8,6))
fig_plt = sns.barplot(x=np.arange(2,15), y=[i['cophenetic'] for i in nmf_grid.values()])
fig_plt.set_xlabel("N-Components")
fig_plt.set_ylabel("Cophenetic")
fig_plt.set_title('Cophenetic Score vs. N-Components')
print(fig_plt)
fig_plt.get_figure().savefig(figure_directory + "NMF/BM25/TagCophenticScores.png", bbox_inches='tight')


fig = plt.figure(figsize=(8,6))
fig_plt = sns.barplot(x=np.arange(2,15), y=[i['rss'] for i in nmf_grid.values()])
fig_plt.set_xlabel("N-Components")
fig_plt.set_ylabel("RSS")
fig_plt.set_title('RSS vs. N-Components')
print(fig_plt)
fig_plt.get_figure().savefig(figure_directory + "NMF/BM25/TagRSSScores.png", bbox_inches='tight')

print("Finished Tag BM25")        


###############################################################################
##COMPANY BM25
###############################################################################

company_vec = GroupVectorizer(tf_type='bm25', apply_idf=True, idf_type='smooth', apply_dl=True, dl_type='linear').fit(trigram_docs, companies)
company_doc_term_matrix = company_vec.transform(trigram_docs, companies)

mod = nimfa.Nmf(V=company_doc_term_matrix, max_iter=200, rank=5)
nmf_grid = nimfa.Nmf.estimate_rank(mod, rank_range=np.arange(2, 6), what=['cophenetic', 'rss'], n_run=10)


fig = plt.figure(figsize=(8,6))
fig_plt = sns.barplot(x=np.arange(2,6), y=[i['cophenetic'] for i in nmf_grid.values()])
fig_plt.set_xlabel("N-Components")
fig_plt.set_ylabel("Cophenetic")
fig_plt.set_title('Cophenetic Score vs. N-Components')
print(fig_plt)
fig_plt.get_figure().savefig(figure_directory + "NMF/BM25/CompanyCophenticScores.png", bbox_inches='tight')


fig = plt.figure(figsize=(8,6))
fig_plt = sns.barplot(x=np.arange(2,6), y=[i['rss'] for i in nmf_grid.values()])
fig_plt.set_xlabel("N-Components")
fig_plt.set_ylabel("RSS")
fig_plt.set_title('RSS vs. N-Components')
print(fig_plt)
fig_plt.get_figure().savefig(figure_directory + "NMF/BM25/CompanyRSSScores.png", bbox_inches='tight')

print("Finished Company BM25")        
                     
###############################################################################
##ANALYST BM25
###############################################################################

analysts_vec = GroupVectorizer(tf_type='bm25', apply_idf=True, idf_type='smooth', apply_dl=True, dl_type='linear').fit(trigram_docs, analysts)
analyst_doc_term_matrix = analysts_vec.transform(trigram_docs, analysts)

mod = nimfa.Lsnmf(V=analyst_doc_term_matrix, max_iter=200, rank=5)
nmf_grid = nimfa.Lsnmf.estimate_rank(mod, rank_range=np.arange(2, 25), what=['cophenetic', 'rss'], n_run=10)


fig = plt.figure(figsize=(8,6))
fig_plt = sns.barplot(x=np.arange(2,25), y=[i['cophenetic'] for i in nmf_grid.values()])
fig_plt.set_xlabel("N-Components")
fig_plt.set_ylabel("Cophenetic")
fig_plt.set_title('Cophenetic Score vs. N-Components')
print(fig_plt)
fig_plt.get_figure().savefig(figure_directory + "NMF/BM25/AnalystCophenticScoresLSNMF.png", bbox_inches='tight')


fig = plt.figure(figsize=(8,6))
fig_plt = sns.barplot(x=np.arange(2,25), y=[i['rss'] for i in nmf_grid.values()])
fig_plt.set_xlabel("N-Components")
fig_plt.set_ylabel("RSS")
fig_plt.set_title('RSS vs. N-Components')
print(fig_plt)
fig_plt.get_figure().savefig(figure_directory + "NMF/BM25/AnalystRSSScoresLSNMF.png", bbox_inches='tight')

print("Finished Analyst BM25 LSNMF")   

###############################################################################
##TAG BM25
###############################################################################

tag_vec = GroupVectorizer(tf_type='bm25', apply_idf=True, idf_type='smooth', apply_dl=True, dl_type='linear').fit(trigram_docs, tags)
tag_doc_term_matrix = tag_vec.transform(trigram_docs, tags)

mod = nimfa.Lsnmf(V=tag_doc_term_matrix, max_iter=200, rank=5)
nmf_grid = nimfa.Lsnmf.estimate_rank(mod, rank_range=np.arange(2, 15), what=['cophenetic', 'rss'], n_run=10)


fig = plt.figure(figsize=(8,6))
fig_plt = sns.barplot(x=np.arange(2,15), y=[i['cophenetic'] for i in nmf_grid.values()])
fig_plt.set_xlabel("N-Components")
fig_plt.set_ylabel("Cophenetic")
fig_plt.set_title('Cophenetic Score vs. N-Components')
print(fig_plt)
fig_plt.get_figure().savefig(figure_directory + "NMF/BM25/TagCophenticScoresLSNMF.png", bbox_inches='tight')


fig = plt.figure(figsize=(8,6))
fig_plt = sns.barplot(x=np.arange(2,15), y=[i['rss'] for i in nmf_grid.values()])
fig_plt.set_xlabel("N-Components")
fig_plt.set_ylabel("RSS")
fig_plt.set_title('RSS vs. N-Components')
print(fig_plt)
fig_plt.get_figure().savefig(figure_directory + "NMF/BM25/TagRSSScoresLSNMF.png", bbox_inches='tight')

print("Finished Tag BM25 LSNMF") 

###############################################################################
##COMPANY BM25
###############################################################################

company_vec = GroupVectorizer(tf_type='bm25', apply_idf=True, idf_type='smooth', apply_dl=True, dl_type='linear').fit(trigram_docs, companies)
company_doc_term_matrix = company_vec.transform(trigram_docs, companies)

mod = nimfa.Lsnmf(V=company_doc_term_matrix, max_iter=200, rank=5)
nmf_grid = nimfa.Lsnmf.estimate_rank(mod, rank_range=np.arange(2, 6), what=['cophenetic', 'rss'], n_run=10)


fig = plt.figure(figsize=(8,6))
fig_plt = sns.barplot(x=np.arange(2,6), y=[i['cophenetic'] for i in nmf_grid.values()])
fig_plt.set_xlabel("N-Components")
fig_plt.set_ylabel("Cophenetic")
fig_plt.set_title('Cophenetic Score vs. N-Components')
print(fig_plt)
fig_plt.get_figure().savefig(figure_directory + "NMF/BM25/CompanyCophenticScoresLSNMF.png", bbox_inches='tight')


fig = plt.figure(figsize=(8,6))
fig_plt = sns.barplot(x=np.arange(2,6), y=[i['rss'] for i in nmf_grid.values()])
fig_plt.set_xlabel("N-Components")
fig_plt.set_ylabel("RSS")
fig_plt.set_title('RSS vs. N-Components')
print(fig_plt)
fig_plt.get_figure().savefig(figure_directory + "NMF/BM25/CompanyRSSScoresLSNMF.png", bbox_inches='tight')

print("Finished Company BM25 LSNMF") 

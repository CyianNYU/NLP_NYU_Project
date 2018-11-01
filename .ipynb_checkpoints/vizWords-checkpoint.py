import numpy as np
import pandas as pd

import re

import spacy
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
nlp = spacy.load('en_core_web_sm')

import string
from collections import Counter
from nltk.corpus import stopwords
stopwords = stopwords.words('english')

punctuations = string.punctuation

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams["xtick.labelsize"] = 7
import seaborn as sns

with open('data/tokenizedQuestions.p', 'rb') as f:
        ec_data = pkl.load(f)

MAX_WORDS = 10

def plotTFIDFbyVar(path, var):
    
    tf_dict = {}

    for v in np.unique([doc[var] for doc in ec_data]):
        if v == "nan":
            continue
        v_docs = [doc['Question'] for doc in ec_data if doc[var]==v]
        usage = Counter(x for xs in v_docs for x in xs)
        tf_dict[v] = {u:usage[u]/np.log2(1 + usage[u]) for u in usage}
        
    tfidf_dict = {}

    for doc, usage in tf_dict.items():
        tfidf_dict[doc] = {}
        total_docs = 0
        for word, value in usage.items():
            for doc2, usage2 in tf_dict.items():
                if word in usage2:
                    total_docs += 1
            tfidf_dict[doc][word] = value*np.log2(1 + len(tf_dict)/total_docs) 
            
    for tag, usage in tfidf_dict.items():
        word_values = sorted(usage.items(), key=lambda kv: -kv[1])[:MAX_WORDS]
        words = [wv[0] for wv in word_values]
        values = [wv[1] for wv in word_values]
        fig = plt.figure(figsize=(12,6))
        fig_plt = sns.barplot(x=words, y=values)
        fig_plt.set_xlabel("Word")
        fig_plt.set_ylabel("TF-IDF Score")
        fig_plt.set_title('TF-IDF Top 10 - {}'.format(tag))
        fig_plt.get_figure().savefig("{}/{}.png".format(path, tag), bbox_inches='tight')
    
    return "Great Success"

print("Visualizing Analyst Corpuses")
plotTFIDFbyVar('figures/analyst_tfidf','Analyst')
print("Visualizing Tag Corpuses")
plotTFIDFbyVar('figures/tag_tfidf','Tag')
print("Visualizing Quarter Corpuses")
plotTFIDFbyVar('figures/quarter_tfidf','Quarter')

#################################################################################
###################################VIZ TAGS BY ANALYST###########################
#################################################################################

df = pd.read_csv("data/origData.csv")

print("Visualizing Analyst/Tag Distributions")
for a, a_d in df.groupby("AnalystName"):
    d_p = a_d.groupby("EarningTag2").size().reset_index(name="Counts")
    fig = plt.figure(figsize=(25,6))
    fig_plt = sns.barplot(x=d_p['EarningTag2'], y=d_p['Counts']/d_p['Counts'].sum())
    fig_plt.set_xlabel("Earning Tag")
    fig_plt.set_ylabel("Count")
    fig_plt.set_title('Earning Tag Distribution - {}'.format(a))
    fig_plt.get_figure().savefig("figures/analyst_eTag/{}.png".format(a), bbox_inches='tight')
    
#################################################################################
###################################VIZ Words By Analyst and Tag##################
#################################################################################
    
def analystTagTfidfPlot():

    groups = df.groupby(['AnalystName', "EarningTag2"])

    freq_dict = {}
    tf_dict = {}

    for j,i in groups:
        words = ' '.join(cleanText(i["Question"])).split()
        usage = Counter(words)
        tf_dict[j] = {u:usage[u]/np.log2(1 + usage[u]) for u in usage}

    tfidf_dict = {}

    for doc, usage in tf_dict.items():
        tfidf_dict[doc] = {}
        total_docs = 0
        for word, value in usage.items():
            for doc2, usage2 in tf_dict.items():
                if doc2[1] == doc[1]:
                    if word in usage2:
                        total_docs += 1
            tfidf_dict[doc][word] = value*np.log2(1 + len(tf_dict)/total_docs) 


    for tag, usage in tfidf_dict.items():
        word_values = sorted(usage.items(), key=lambda kv: -kv[1])[:10]
        words = [wv[0] for wv in word_values]
        values = [wv[1] for wv in word_values]
        fig = plt.figure(figsize=(12,6))
        fig_plt = sns.barplot(x=words, y=values)
        fig_plt.set_xlabel("Word")
        fig_plt.set_ylabel("TF-IDF Score")
        fig_plt.set_title('TF-IDF Top 10 - {}-{}'.format(tag[0], tag[1]))
        fig_plt.get_figure().savefig("{}/{}_{}.png".format('figures/analyst_eTag_tfidf', tag[0],tag[1]), bbox_inches='tight')

print("Visualizing Analyst/Tag TFIDF")
analystTagTfidfPlot()
print("Finished")
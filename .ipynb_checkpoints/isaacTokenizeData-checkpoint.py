import pickle as pkl
import re

import numpy as np
import pandas as pd

from gensim.models.phrases import Phrases, Phraser

import textacy
from textacy import preprocess_text, Doc, Corpus
from textacy.vsm import Vectorizer, GroupVectorizer
from textacy.tm import TopicModel
en = textacy.load_spacy("en_core_web_sm", disable='parser')

data = pd.read_csv('data/qaData.csv', parse_dates=['Date'])
ec_data = data.loc[data['EventType']=="Earnings call", ['Date', 'Company', 'Participants', 'AnalystName',	'AnalystCompany', 'EventName', 'EarningTag2', "Question"]].copy()
ec_data['Quarter'] = ec_data['EventName'].str.split("Q").str[0]
ec_data = ec_data.groupby(['Date', "Company", "Participants", "EventName", "Quarter"]).apply(lambda x: x.reset_index()).reset_index(drop=True)
ec_data.columns = ["QuestionOrder", "Date", "Company", "Participants", "AnalystName", "AnalystCompany", "EventName", "Tag", "Question", "Quarter"]
ec_data = ec_data[["Date", "Quarter", "Company", "Participants", "AnalystCompany", "AnalystName", "QuestionOrder", "Tag", "Question"]]

docs = Corpus(lang=en, docs=ec_data.apply(lambda x: Doc(content=' '.join(
                                                        [token for token in preprocess_text(text=x['Question'], lowercase=True, no_punct=True, no_contractions=True, no_accents=True, no_currency_symbols=True, no_numbers=True).split(' ') if len(token)>2]),
                                                    lang=en, metadata={'Quarter':x['Quarter'],
                                                                       'Company':x['Company'],
                                                                       'QuestionOrder':x['QuestionOrder'],
                                                                       'Analyst':x["AnalystName"],
                                                                       'Tag':x['Tag']}),axis=1).tolist())
tokenized_docs = [list(doc.to_terms_list(ngrams=(1), as_strings=True, normalize='lemma', drop_determiners=True)) for doc in docs]

bigram_phraser = Phraser(Phrases(tokenized_docs, min_count=10, threshold=25, delimiter=b' '))
bigram_docs = [bigram_phraser[doc] for doc in tokenized_docs] 

trigram_phraser = Phraser(Phrases(bigram_docs, min_count=5, threshold=10, delimiter=b' '))
trigram_docs = [trigram_phraser[doc] for doc in bigram_docs]

ec_list = [{'Quarter':docs[i].metadata['Quarter'], 
            'Company':docs[i].metadata['Company'], 
            'QuestionOrder':docs[i].metadata['QuestionOrder'], 
            'Analyst':docs[i].metadata['Analyst'], 
            'Tag':docs[i].metadata['Tag'], 
            'Question':trigram_docs[i]} for i in range(len(trigram_docs))]

with open('data/tokenizedQuestions.p', 'wb') as f:
    pkl.dump(ec_list, f)

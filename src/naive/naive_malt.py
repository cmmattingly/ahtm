import os
os.chdir(os.environ['PROJECT_DIR'])

from graphviz import Source
from datasets import load_dataset # hugging face datasets
from tqdm import tqdm
import pickle
import pandas as pd
import numpy as np

import nltk
try:
    nltk.corpus.stopwords.words('english')
except LookupError:
    nltk.download('stopwords') # download nltk stopwords
    
import gensim.parsing.preprocessing as gsp
from gensim import utils
import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel
    
# from nltk.parse.malt import MaltParser 
from malt.malt import MaltParser # source code from nltk library
from lda.LDAMallet import LdaMallet # source code for gensim LDA (gibbs sampling) mallet wrapper 

import naive_stanford

def reduce_parser(parser: any) -> list[any]:
    ''' Reduce iter of iters into list of dependency trees'''
    return [next(list_it) for list_it in parser]

def tokenize_doc(doc: str) -> list[list[str]]:
    ''' Tokenize document into sentences represented as tokens '''
    return [nltk.word_tokenize(sent) for sent in nltk.sent_tokenize(doc)]

def tag_sents(sents: list[list[str]]) -> list[list[str]]:
    ''' POS tag sentences '''
    return list(map(pos_tag, sents))

def get_dependency_trees(docs: str, malt_parser_version='maltparser-1.7.2', model_version='engmalt.linear-1.7.mco') -> list[list[any]]:
    ''' Calculate dependency relation trees using malt parser '''
    # initalize malt parser model
    mp = MaltParser(malt_parser_version, model_version, tagger=nltk.pos_tag)
    
    # create <doc_idx, tokenized_sent> list of sents
    sents = [
        (i, nltk.word_tokenize(sent))
        for i, doc in enumerate(docs)
        for sent in nltk.sent_tokenize(utils.to_unicode(str(doc).lower())) # convert doc to lowercase, and sentence tokenized.
    ]
      
    # unzip list of tuples
    doc_idxs, sents = zip(*sents)
    
    # create parser <generator> and loop through parser to produce dependency tree for each sentence
    parser = mp.parse_sents(sents, verbose=True)

    # define valid word
    valid_word = lambda word: not word in stop_words and word.isalpha() and len(word) > 2
    # initialize document hashmap
    doc_reln_pairs = {i:[] for i in set(doc_idxs)}

    i = 0
    for list_it in tqdm(parser):
        tree = next(list_it)
        try:
            tree.tree()
        except:
            doc_reln_pairs[doc_idxs[i]].extend(naive_stanford.get_tokens(text))
            continue

        for gov, reln, dep in tree.triples():
        # [NOTE]: No londer needed due to pre-processing already occuring
        #     # if not valid_word(gov[0]) or not valid_word(dep[0]):
        #     #     continue

            doc_reln_pairs[doc_idxs[i]].extend([f"{gov[0]}{sep}{reln}.gov", f"{dep[0]}{sep}{reln}.dep"])

        i += 1
    
    return doc_reln_pairs

def get_topics(model: any, n_topics: int) -> dict:
    '''Returns dictionary of topics'''
    
    topics_dict = dict(model.print_topics(num_topics=n_topics))
    topics_dict = {int(k):v for k,v in topics_dict.items()}
    
    return topics_dict

def get_doc_top_matrix(model):
    doc_top_matrix = [*model.load_document_topics()]

    expected_keys = list(range(10))
    new_doc_top_matrix = []
    for doc_top in tqdm(doc_top_matrix):
        _dict = dict(doc_top)
        for key in expected_keys:
            if key not in _dict:
                _dict[key] = 0
        new_doc_top_matrix.append(list(_dict.items()))

    doc_top_matrix = [sorted(arr) for arr in new_doc_top_matrix]
    for i in range(len(doc_top_matrix)):
        doc_top_matrix[i] = np.array([tpl[1] for tpl in doc_top_matrix[i]])

    return np.array(doc_top_matrix)

def get_top_docs(doc_top_matrix, n_topics=10, k=10):
    for i in range(n_topics):
        print(f"Topic: {i} -------------------------------------------------")
        idxs = np.argsort(doc_top_matrix[:,i])[-k:]
        for idx in idxs:
            print("---------------")
            print(doc_top_matrix[idx, i], texts[idx])
            print("---------------")

import os
os.chdir(os.environ['PROJECT_DIR'])

from datasets import load_dataset # hugging face datasets
from tqdm import tqdm
import pickle
import pandas as pd
import numpy as np
import argparse
    
import gensim.parsing.preprocessing as gsp
from gensim import utils
import gensim.corpora as corpora
    
# from nltk.parse.malt import MaltParser 
from malt.malt import MaltParser # source code from nltk library
from lda.LDAMallet import LdaMallet # gensim LDA (gibbs sampling) mallet wrapper 

import naive_malt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', nargs='?', type=str, default='data/data.csv')
    parser.add_argument('--n_topics', nargs='?', type=int, default=40)
    parser.add_argument('--n_top_docs', nargs='?', type=int, default=40)
    args = parser.parse_args()

    return args

def main():
    # parse passed arguments
    args = parse_args()
    path = args.path
    n_topics = args.n_topics
    n_top_docs = args.n_top_docs

    # read text data
    # df = pd.read_csv(path)
    # df['text'] = df['text'].astype('string')
    # texts = list(df['text'])
    dailymail = load_dataset('cnn_dailymail', '2.0.0') # https://huggingface.co/datasets/cnn_dailymail/viewer/2.0.0/
    texts = dailymail['train']['article'][:200]

    # get grammatical relations pairs 
    doc_reln_pairs = naive_malt.get_dependency_trees(texts)
    tokens = list(doc_reln_pairs.values())

    # run LDA
    id2word = corpora.Dictionary(tokens)
    corpus = list(map(lambda x: id2word.doc2bow(x), tokens))

    model = LdaMallet(os.environ['MALLET_DIR'], corpus=corpus, num_topics=n_topics, id2word=id2word)

    # get document x topic matrix
    doc_top_matrix = naive_malt.get_doc_top_matrix(model)

    # print topics and top documents   
    naive_malt.get_topics(model, n_topics=n_topics) 
    naive_malt.get_top_docs(doc_top_matrix, n_topics=n_topics, k=n_top_docs)

    # save document x topic matrix
    with open('data/doc_top_matrix.npy', 'wb') as f:
        np.save(f, doc_top_matrix)
    
    # save lda mallet model
    with open('models/lda_gr.pkl', 'wb') as f:
        pickle.dump(model, f) 

    # output: printed
    # topic: [(word, prob)]
    # list of top topic documents - 20 document
    # stored:
    # doc_top_matrix
    # model

if __name__ == "__main__":
    main()
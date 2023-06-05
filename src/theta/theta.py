import numpy as np
import json
import argparse
import os
import random
random.seed(2023)

import gensim.corpora as corpora
from word import Word
from tqdm import tqdm

from naive.naive_malt import get_dependency_trees

def parse_args():
    parser = argparse.ArgumentParser()

    # n topics and n latent theta roles
    parser.add_argument('--K', nargs='?', type=int, default=5)
    parser.add_argument('--T', nargs='?', type=int, default=2)

    # dirichlet initialization hyper parameters (static)
    parser.add_argument('--alpha', nargs='?', type=float, default=0.1) 
    parser.add_argument('--eta', nargs='?', type=float, default=0.1) 
    parser.add_argument('--gamma', nargs='?', type=float, default=0.1) 
    parser.add_argument('--lam', nargs='?', type=float, default=0.1)
    parser.add_argument('--n_iters', nargs='?', type=int, default=20)
    parser.add_argument('--corpus_path', nargs='?', type=str, default="data/processed/corpus.json")

    args = parser.parse_args()

    return args.K, args.T, args.alpha, args.eta, args.gamma, args.lam, args.n_iters, args.corpus_path
 
def main():
    K, T, alpha, eta, gamma, lam, n_iters, corpus_path = parse_args()

    # [TODO]: change to BSON instead of JSON for faster io and smaller storage
    with open(corpus_path) as json_file:
        o = json.load(json_file)

    doc_objects = o['documents']
    docs = [ doc_objects[str(doc_id)]['words'] for doc_id in doc_objects ]
    doc_relns = [ doc_objects[str(doc_id)]['relns'] for doc_id in doc_objects ]
    vocab = o['vocab']
    global_relns = o['vocab_relns']

    # document preprocessing helpers
    id2word = corpora.Dictionary(docs)
    reln2id = {reln:i for i, reln in enumerate(global_relns)}
    corpus = list(map(lambda x: id2word.doc2idx(x), docs))

    # initialize scalars from plate diagram
    N = 0 # n words of document
    D, V, R = len(docs), len(vocab), len(global_relns) # n documents, n words, n relns: https://universaldependencies.org/u/dep/
    
    # initialize matricies from plate diagram
    theta = np.zeros((D, K)) # documents x topic matrix 
    phi = np.zeros((D, T)) # documents x theta roles matrix
    beta = np.zeros((K, V)) # topics x vocabulary
    zeta = np.zeros((T, R)) # theta roles x grammatical relationships

    # counts for gibbs sampling 
    # -- topics
    n_d_k = np.zeros((D, K))
    n_k_w = np.zeros((K, V))
    n_k = np.zeros(K)
    # -- theta
    n_d_t = np.zeros((D, T))
    n_t_k = np.zeros((T, K))
    n_t_reln = np.zeros((T, R))
    n_t = np.zeros(T)

    # https://ethen8181.github.io/machine-learning/clustering_old/topic_model/LDA.html
    # https://sandipanweb.wordpress.com/2020/12/07/bayesian-machine-learning-mcmc-and-probabilistic-programming-with-python/

    # initalization loop and word object creation
    print('Initializing variables...')
    corpus_hat = []
    for d, doc in enumerate(corpus):
        doc_hat = []
        for i, word_id in enumerate(doc):
            # initalize topics
            z = np.random.randint(K) # random topic
            y = np.random.randint(T) # random theta role

            relns = doc_relns[d][i]
            if not relns:
                continue
            
            for reln in relns:
                n_t_reln[y, reln2id[reln]] += 1

            n_d_t[d, y] += 1
            n_t_k[y, z] += 1
            n_t[y] += 1

            word = Word(content=id2word[word_id], idx=word_id, z=z, y=y, relns=relns)
            doc_hat.append(word)

            # initalize counts
            n_d_k[d, z] += 1
            n_k_w[z, word_id] += 1
            n_k[z] += 1
        
        corpus_hat.append(doc_hat)

    # gibbs sampling
    print('Running gibbs sampling...')
    for i in tqdm(range(n_iters)):
        for d, doc in enumerate(corpus_hat):
            N = len(doc)

            for word in doc:
                # obtain topic and word id
                z, y, idx, relns = word.z, word.y, word.idx, word.relns
            
                # update counts
                n_d_k[d, z] -= 1
                n_k_w[z, idx] -= 1
                n_k[z] -= 1

                for reln in relns:
                    n_t_reln[y, reln2id[reln]] -= 1

                n_d_t[d, y] -= 1
                n_t_k[y, z] -= 1
                n_t[y] -= 1

                # calculate conditional probability distribution for topic selection
                p_k = np.zeros(K)
                for k in range(K):
                    # p(k | d) = (num_docs_for_topic + alpha) / (num_words_in_doc + num_topics * alpha)
                    p_k_d = (n_d_k[d, k] + alpha) / (N + K * alpha)

                    # p(w | k) = (num_curr_topic_for_word + eta) / ()
                    p_w_k = (n_k_w[k, idx] + eta) / (n_k[k] + eta * V)
                    p_k[k] = p_k_d * p_w_k

                # select topic
                z = np.random.choice(K, p=p_k / sum(p_k))
                word.z = z

                # update counts
                n_d_k[d, z] += 1
                n_k_w[z, idx] += 1
                n_k[z] += 1

                p_t_reln = np.zeros((T, R))
                for reln in relns:
                    for t in range(T):
                        p_t_d = (n_d_t[d, t] + gamma) / (N + T * gamma)
                        p_reln_t = (n_t_reln[t, reln2id[reln]] + lam) / (n_t[t] + lam * R)

                        p_z_d = (n_d_k[d, z] + alpha) / (N + K * alpha) 
                        p_z_t = (n_t_k[t, z] + gamma) / (n_t[t] + gamma * T)
                        p_z_d_t = p_z_d * p_z_t
                        
                        p_t_reln[t, reln2id[reln]] = p_t_d * p_reln_t * p_z_d_t
                
                # select theta role for each relations
                p_t = np.sum(p_t_reln, axis=1)
                y = np.random.choice(T, p=p_t / np.sum(p_t))
                word.y = y

                # update counts
                for reln in relns:
                    n_t_reln[y, reln2id[reln]] += 1

                n_d_t[d, y] += 1
                n_t_k[y, z] += 1
                n_t[y] += 1

    # compute beta matrix
    for k in range(K):
        beta[k] = (n_k_w[k] + eta) / (n_k[k] + V * eta)

    # compute theta matrix
    for d in range(D):
        N = len(corpus[d])
        theta[d] = (n_d_k[d] + alpha) / (N + K * alpha)

    # compute zeta matrix
    for t in range(T):
        zeta[t] = (n_t_reln[t] + lam) / (n_t[t] + R * lam)

    # compute phi matrix
    for d in range(D):
        N = len(corpus[d])
        phi[d] = (n_d_t[d] + gamma) / (N + T * gamma)
     
    # ------ printing --------
    for k in range(K):
        word_ids = np.argsort(beta[k])[::-1][:10]
        probs = np.sort(beta[k])[::-1][:10]
        top_words = [id2word[i] for i in word_ids]
        strings = [f'{prob} * {word}' for prob, word in zip(probs, top_words)]
        print(f"Topic {str(k)}: {', '.join(strings)} \n")
    
    print("\n")

    for t in range(T):
        reln_ids = np.argsort(zeta[t])[::-1][:10]
        probs = np.sort(zeta[t])[::-1][:10]
        top_relns = [global_relns[i] for i in reln_ids]
        strings = [f'{prob} * {reln}' for prob, reln in zip(probs, top_relns)]
        print(f"Theta Role {str(t)}: {', '.join(strings)}\n")
    
    print("\n")

    for d in range(D):
        top_topics = np.argsort(theta[d])[::-1][:3]
        top_theta = np.argsort(phi[d])[::-1][:3]
        print(f"Document {str(d)}: {' '.join(str(top_topics))} -- {' '.join(str(top_theta))}")

if __name__ == "__main__":
    main()

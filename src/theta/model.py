import numpy as np
from tqdm import tqdm
from word import Word

class ThetaRoleModel:
    def __init__(self, corpus, cfs, doc_relns, corpus_relns, id2word, reln2id, n_iters, K, T, D, V, R, alpha, eta, gamma, lam):
        self.corpus = corpus # corpus of word ids
        self.cfs = cfs # count frequencies for words
        self.doc_relns = doc_relns # relns in documents
        self.corpus_relns = corpus_relns # relns in form of vocab
        self.id2word = id2word # convert word id to word
        self.reln2id = reln2id # convert reln to id
        self.corpus_hat = None # corpus in the form of Word objects
       
        self.n_iters = n_iters

        # n topics, n theta roles, n documents, n words, n relations: https://universaldependencies.org/u/dep/
        self.K = K
        self.T = T
        self.D = D
        self.V = V
        self.R = R

        # parameters for dirichlet distribution(s)
        self.alpha = alpha
        self.eta = eta
        self.gamma = gamma
        self.lam = lam

        # matrices from plate diagram
        self.theta = None # documents x topic matrix 
        self.phi = None # documents x theta roles matrix
        self.beta = None # topics x vocabulary
        self.zeta = None # theta roles x grammatical relationships
        
        # counts for gibbs samplins
        self.n_d_k = None
        self.n_k_w = None
        self.n_k = None
        self.n_d_t = None
        self.n_t_k = None
        self.n_t_reln = None
        self.n_t = None


    def initialize_variables(self):
        '''
        Initialize Word objects to store a random topic and theta role for gibbs sampling
        '''
        # initalize counts for gibbs sampling
        self.n_d_k = np.zeros((self.D, self.K))
        self.n_k_w = np.zeros((self.K, self.V))
        self.n_k = np.zeros(self.K)
        self.n_d_t = np.zeros((self.D, self.T))
        self.n_t_k = np.zeros((self.T, self.K))
        self.n_t_reln = np.zeros((self.T, self.R))
        self.n_t = np.zeros(self.T)

        self.corpus_hat = []
        for d, doc in enumerate(self.corpus):
            doc_hat = []
            for i, word_id in enumerate(doc):
                z = np.random.randint(self.K) # random topic
                y = np.random.randint(self.T) # random theta role

                # obtain relns for related word in current document
                relns = self.doc_relns[d][i]
                if not relns:
                    continue
                
                # initalize counts
                self.n_d_k[d, z] += 1
                self.n_k_w[z, word_id] += 1
                self.n_k[z] += 1

                for reln in relns:
                    self.n_t_reln[y, self.reln2id[reln]] += 1
                self.n_d_t[d, y] += 1
                self.n_t_k[y, z] += 1
                self.n_t[y] += 1

                word = Word(content=self.id2word[word_id], idx=word_id, z=z, y=y, relns=relns)
                doc_hat.append(word)

            self.corpus_hat.append(doc_hat)

    def fit(self):
        '''
        Run gibbs sampling for n iterations to assign theta role and topic to each word
        '''
        for i in tqdm(range(self.n_iters)):
            for d, doc in enumerate(self.corpus_hat):
                N = len(doc)

                for word in doc:
                    z, y, idx, relns = word.z, word.y, word.idx, word.relns
                
                    # subtract count for current topic and theta role
                    self.n_d_k[d, z] -= 1
                    self.n_k_w[z, idx] -= 1
                    self.n_k[z] -= 1
                    for reln in relns:
                        self.n_t_reln[y, self.reln2id[reln]] -= 1
                    self.n_d_t[d, y] -= 1
                    self.n_t_k[y, z] -= 1
                    self.n_t[y] -= 1

                    # calculate conditional probability distribution for topic selection
                    # p(k) = p(k | d) * p(w | k)
                    p_k = np.zeros(self.K)
                    for k in range(self.K):
                        p_k_d = (self.n_d_k[d, k] + self.alpha) / (N + self.K * self.alpha)
                        p_w_k = (self.n_k_w[k, idx] + self.eta) / (self.n_k[k] + self.eta * self.V)
                        p_k[k] = p_k_d * p_w_k

                    # select topic
                    z = np.random.choice(self.K, p=p_k / sum(p_k))
                    word.z = z

                    # update counts for selected topic
                    self.n_d_k[d, z] += 1
                    self.n_k_w[z, idx] += 1
                    self.n_k[z] += 1

                    # probability of current topic given current grammatical relations
                    # p(t | reln) = p(t | d) * p(reln | t) * p(z | d,t)
                    p_t_reln = np.zeros((self.T, self.R))
                    for reln in relns:
                        for t in range(self.T):
                            p_t_d = (self.n_d_t[d, t] + self.gamma) / (N + self.T * self.gamma)
                            p_reln_t = (self.n_t_reln[t, self.reln2id[reln]] + self.lam) \
                                 / (self.n_t[t] + self.lam * self.R)

                            p_z_d = (self.n_d_k[d, z] + self.alpha) / (N + self.K * self.alpha) 
                            p_z_t = (self.n_t_k[t, z] + self.gamma) \
                                 / (self.n_t[t] + self.gamma * self.T)
                            p_z_d_t = p_z_d * p_z_t
            
                            p_t_reln[t, self.reln2id[reln]] = p_t_d * p_reln_t * p_z_d_t
                    
                    # select theta role for each relation using probability distribution
                    p_t = np.sum(p_t_reln, axis=1)
                    y = np.random.choice(self.T, p=p_t / np.sum(p_t))
                    word.y = y

                    # update counts for selected theta role
                    for reln in relns:
                        self.n_t_reln[y, self.reln2id[reln]] += 1
                    self.n_d_t[d, y] += 1
                    self.n_t_k[y, z] += 1
                    self.n_t[y] += 1

    def compute_theta(self):
        '''
        Compute the theta matrix from the plate diagram
        '''
        self.theta = np.zeros((self.D, self.K))
        for d in range(self.D):
            N = len(self.corpus[d])
            self.theta[d] = (self.n_d_k[d] + self.alpha) / (N + self.K * self.alpha)
        return self.theta

    def compute_phi(self):
        '''
        Compute the phi matrix from the plate diagram
        '''
        self.phi = np.zeros((self.D, self.T))
        for d in range(self.D):
            N = len(self.corpus[d])
            self.phi[d] = (self.n_d_t[d] + self.gamma) / (N + self.T * self.gamma) 
        return self.phi

    def compute_beta(self):
        '''
        Compute the beta matrix from the plate diagram
        '''
        self.beta = np.zeros((self.K, self.V))
        for k in range(self.K):
            self.beta[k] = (self.n_k_w[k] + self.eta) / (self.n_k[k] + self.V * self.eta)
        return self.beta

    def compute_zeta(self):
        '''
        Compute the zeta matrix from the plate diagram
        '''
        self.zeta = np.zeros((self.T, self.R))
        for t in range(self.T):
            self.zeta[t] = (self.n_t_reln[t] + self.lam) / (self.n_t[t] + self.R * self.lam) 
        return self.zeta

    # p(reln|w) = p(reln|y)p(y|z)p(z|w)
    def compute_reln_w(self, k, t, reln, w,):
        p_reln_y = (self.n_t_reln[t, self.reln2id[reln]] + self.lam) \
            / (self.n_t[t] + self.lam * self.R)
        p_y_z = (self.n_t_k.T[k, t] + self.gamma) \
            / (self.n_k[k] + self.gamma * self.K)
        p_z_w = (self.n_k_w.T[w, k] + self.eta) / (self.cfs(w) + self.eta * self.V)
        p_reln_w = p_reln_y * p_y_z * p_z_w

        return p_reln_w

    def print_topics(self):
        '''
        Print topics and the assigned theta role 
        '''
        n_k_t = self.n_t_k.T
        for k in range(self.K):
            # select theta role for given topic
            prob_distribution = n_k_t[k] / sum(n_k_t[k])
            y = np.random.choice(self.T, p=prob_distribution)

            # obtain top words for each topic
            word_ids = np.argsort(self.beta[k])[::-1][:10]
            probs = np.sort(self.beta[k])[::-1][:10]
            top_words = [self.id2word[i] for i in word_ids]
            strings = [f'{prob} * {word}' for prob, word in zip(probs, top_words)]
            print(f"Topic {str(k)} (Theta Role: {str(y)}): {', '.join(strings)} \n") 

    def print_theta_roles(self):
        '''
        Print theta roles
        '''
        for t in range(self.T):
            reln_ids = np.argsort(self.zeta[t])[::-1][:10]
            probs = np.sort(self.zeta[t])[::-1][:10]
            top_relns = [self.corpus_relns[i] for i in reln_ids]
            strings = [f'{prob} * {reln}' for prob, reln in zip(probs, top_relns)]
            print(f"Theta Role {str(t)}: {', '.join(strings)}\n")

    def print_top_of_docs(self):
        '''
        Print top topics and theta roles for each document
        '''
        for d in range(self.D):
            top_topics = np.argsort(self.theta[d])[::-1][:3]
            top_theta = np.argsort(self.phi[d])[::-1][:3]
            print(f"Document {str(d)}: {' '.join(str(top_topics))} -- {' '.join(str(top_theta))}")
    
    def compute_matrices(self):
        _theta = self.compute_theta()
        _phi = self.compute_phi()
        _beta = self.compute_beta()
        _zeta = self.compute_zeta()

        return _theta, _phi, _beta, _zeta

    def print_all(self):
        self.print_topics()
        self.print_theta_roles()
        self.print_top_of_docs()
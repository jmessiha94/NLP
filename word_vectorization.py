import numpy as np
import pandas as pd
import re

import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

# tokenization
from nltk.tokenize import word_tokenize

from pre_processing import pre_process

def word_embedding(X):
    # Returns each sentence with a 'set' of words and their word count
    embedded = []
    for sent in X:
        set_sent = set(sent)
        new_sent = {}
        for word in set_sent:
            new_sent[word] = sent.count(word)
        embedded.append(new_sent)
    return embedded
        
def bag_of_words(X):
    # X is tokenized and preprocessed
    combined_X = []
    for x in X:
        combined_X += x
        
    combined_X = sorted(list(set(combined_X)))
    embeddings = word_embedding(X)

    bag_o_words = []
    for sentence in embeddings:
        embedded = []
        for i in combined_X:
            try:
                j = int(sentence[i])
            except:
                j = 0
            embedded.append(j)
        bag_o_words.append(embedded)
    return np.array(bag_o_words)



def doc2vec_create_model(X,max_epochs,vec_size,alpha,min_alpha=0.00025,min_count=1,dm=1,save=None):
    
    tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(X)]

    model = Doc2Vec(vector_size=vec_size,
                    alpha=alpha, 
                    min_alpha=0.00025,
                    min_count=1,
                    dm =1)    
    
    model.build_vocab(tagged_data)
    
    for epoch in range(max_epochs):
#        print('iteration {0}'.format(epoch))
        model.train(tagged_data,
                    total_examples=model.corpus_count,
                    epochs=model.iter)
        # decrease the learning rate
        model.alpha -= 0.0002
        # fix the learning rate, no decay
        model.min_alpha = model.alpha
    if save != None:
        model.save(save)
    
    vectorized_X = []
    for i in range(len(X)):
        vectorized_X.append(model.docvecs[str(i)])
    
    return model,vectorized_X


def doc2vec_use_model(X,model):
    vectorized_X = []
    for x in X:
        vectorized_X.append(model.infer_vector(x))
        
    return vectorized_X



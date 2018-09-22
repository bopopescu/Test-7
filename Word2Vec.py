# future  = Missing Link betwwen Python 2/3 allows to use syntax from both
#codecs = for word encoding
#glob = for regex for files search
#re = for rgex at granular level
#multiprocessing = for multithreading/Concurrency
#os =dealing with files mainly Operating System
#pPrint = for betiful Print making it User readable
#nltk = Natural Language Processing
#WebSummarizer = custom Module For getting Data
#w2v=google created words Vector collection
#sklearn.manifold = dimesion reduction of word vector for sze 300/500 to 2/3
#numpy = Maths Library Helps in above reduction
#matp= for graph Plotting
#pandas = Parser Helping
#seaborn= for visualising final data set
from __future__ import absolute_import,division, print_function
import codecs
import glob
import re
import multiprocessing
import os
import pprint
import nltk
import gensim
import gensim.models.word2vec as w2v
import sklearn.manifold
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import WebSummarizerNltk

def sentence_to_wordlist(raw):
    words=raw.split();
    return words


#get raw data from Previous Task filtered Organised Preprocessed Text
rawText = WebSummarizerNltk._getRawData()
print ('RawData')
print (rawText)
print('ProcessedData')
#Tokenising Raw Data
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
raw_sentences=tokenizer.tokenize(rawText)
sentences=[]

#getting wrd token List tokenised Data
for raw_sentence in raw_sentences:
    if len(raw_sentence)>0:
        sentences.append(sentence_to_wordlist(raw_sentence))

#just temporary to find count no use in code
tokenCount=sum([len(sentence)for sentence in sentences])
print('Token Count----')
print(tokenCount)

#Initialise defaault Pram for -Word 2 Vec. It helps Distance, Similarity,Ranking create w2v

#numfeatures=Dimesion Count. More dimension more complex  to train but more accurate.
#min_word_count=smmallest set of word  want to consider for vector
#num_of_worker= number cpu process we want run parralllel in concurancy
#context_size = size of block of works
#downsampling = setting sample for frequent words more ocuuring will be used less for creating vector 0 - 1e-5
#seed= random no generator


num_features=200
min_word_count=3
num_of_worker=multiprocessing.cpu_count()
context_size =7
downsampling = 1e-3
seed= 1


#buildModel
thrones2Vec = w2v.Word2Vec(sg=1,seed=seed,workers=num_of_worker,size=num_features,min_count=min_word_count,window=context_size,sample=downsampling)
thrones2Vec.build_vocab(sentences)
print('WordVecCount----')
print(thrones2Vec)
thrones2Vec.train(sentences, epochs=thrones2Vec.iter, total_examples=thrones2Vec.corpus_count)


#save the trained Model for future
if not os.path.exists("trained"):
    os.makedirs("trained")
thrones2Vec.save(os.path.join("trained","thrones2Vec.w2v"))


#thrones2Vec =gensim.models.KeyedVectors.load_word2vec_format('ANY_MODEL.bin.gz',os.path.join("trained","thrones2Vec.w2v"),binary=False,datatype=bytes)
thrones2Vec = w2v.Word2Vec.load(os.path.join("trained","thrones2Vec.w2v"))


#Comprees 300 dimension to 2dimension Vector using  tsne=t statistic distributed neighbour embeded
#how to visualise dataset easily video for this
tnse = sklearn.manifold.TSNE(n_components=2,random_state=0)
all_word_vectors_matrix = thrones2Vec.syn1neg
all_word_vectors_matrix_2d = tnse.fit_transform(all_word_vectors_matrix)

points = pd.DataFrame(
    [
        (word,coords[0],coords[1])
        for word, coords in[
            (word,all_word_vectors_matrix_2d[thrones2Vec.vocab[word].index])
            for word in thrones2Vec.wv.iter
      ]
    ],columns=["word","x","y"]
)









print(sentences)




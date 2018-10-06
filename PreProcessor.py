from collections import defaultdict
from nltk.corpus import stopwords
from string import punctuation
from nltk.tokenize import sent_tokenize, word_tokenize
import re
import heapq
import os
class WebSummarizerNltkClass:
    def _init_(self):
        print 'hello'

    def _organiseData(self,text,title):

        sentences = sent_tokenize(text)
        wordSent = [word_tokenize(s.lower()) for s in sentences]
        sFreq=self._compute_frequencies(wordSent)
        ranking=defaultdict(int)
        for i,sentence in enumerate(wordSent):
            for word in sentence:
                if word in sFreq:
                    ranking[i]+=1
        sentences_index=heapq.nlargest(len(ranking),ranking)
        return [sentences[j] for j in sentences_index]

    def _compute_frequencies(self,wordSent,customStopWrds=None):
        freq=defaultdict(int)
        if customStopWrds is None:
            stpWords = set(stopwords.words("english")+list(punctuation))
        else:
            stpWords = set((customStopWrds).Union(stopwords.words("english")+list(punctuation)))
        for sentence in wordSent:
            for word in sentence:
                if word not in stpWords:
                    freq[word] +=1
        m=float(max(freq.values()))
        for word in freq.keys():
            freq[word]=freq[word]/m
            if freq[word]<=0.1 or freq[word]>=0.9:
                del freq[word]
        return freq

def filterStopWords(data):
    stop_words = set(stopwords.words("english"))
    words = word_tokenize(data)
    filtered_sent=''
    for w in words:
        if w not in stop_words:
            filtered_sent.__add__(w)
    return filtered_sent



def preProcessData(data):
  #filteredData= filterStopWords(data)
  obj=WebSummarizerNltkClass()
  summaryArray =obj._organiseData(data,data)
  finalSummary =""
  for sentence in summaryArray:
     finalSummary=finalSummary+sentence
  return finalSummary

#preProcessData('My Name Is ujwal Singh. My Age is 18. I work In London. I am Indian Citizen')






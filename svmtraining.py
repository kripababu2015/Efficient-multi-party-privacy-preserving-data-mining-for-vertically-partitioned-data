##from functools import reduce
##
##from sklearn import cross_validation, metrics, ensemble
##import numpy as np
##
##from skfusion import datasets
##from skfusion import fusion as skf
import re
import unicodedata
import json
from sklearn import model_selection
import pandas as pd 
from sklearn.cross_decomposition import PLSCanonical
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import numpy as np
# reading csv file  
data=pd.read_csv("csvfile.csv")

dd=[]
tdata=[]
print(len(data))
for i in range(len(data)):
        l=[]
        #print(data["Source "][i],data["Traffic"][i])
        s=data["Summary"][i]
        v=data["Placed"][i]
        

        dd.append(s)
        tdata.append(v)
print(len(dd))
print(len(tdata))
print(dd[0])
print(tdata)

import nltk
import pandas as pd
import numpy as np
import time
import os
import pickle
from sklearn.svm import LinearSVC
#protocol;src_bytes;ds_bytes;same_srv;diff_srv_rate;dst_host_srv_count;dst_host_same_srv_rate;dst_host_diff_srv_rate;dst_host_same_src_port_rate;dst_host_rerror_rate;dst_host;count;dst_host_count;dst_host_srv_count;dst_host_srv_serror_rate;cl;label

# get the word lists of sentences
def get_words_in_sentences(sentences):
            all_words = []
            for (words, sentiment) in sentences:
                    all_words.extend(words)
            return all_words

def close(wndw):
        wndw.destroy()

def result(result):
        rs=result
        print (rs[0])
        print ("-------------------------")

# get the unique word from the word list
def get_word_features(wordlist):
            wordlist = nltk.FreqDist(wordlist)
            word_features = wordlist.keys()
            return word_features


def testit(sente_tests):

    word_features = get_word_features(sente_tests)
    def extract_features(document):
        document_words = set(document)
        features = {}
        for word in word_features:
          features['contains(%s)' % word] = (word in document_words)
        return features
    if "https" in sente_tests:
        return 0
    else:
        f=open("myclass.pickle",'rb')
        classi=pickle.load(f)
        emot=classi.classify(extract_features(sente_tests.split()))
        print ("--> ",emot)
        return emot
def mainn():
    train = pd.read_csv("csvfile.csv", header=0,delimiter=",", quoting=1)
    num_reviews = train["Placed"].size
    print (num_reviews)
    
    data=[]
    sentiments=[]
    global sentences
    sentences = []
    print (os.getcwd())
    for i in range( 0,num_reviews ):
        sente = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','',train["Summary"][i])
        sente = re.sub('@[^\s]+','',sente)
        #Remove additional white spaces
        sente = re.sub('[\s]+', ' ', sente)
        #Replace #word with word
        sente = re.sub(r'#([^\s]+)', r'\1', sente)
        #trim
        sente = sente.strip('\'"')
        words_filtereds = [e.lower() for e in sente.split() if len(e) >= 3]
        sentences.append((words_filtereds,train["Summary"][i]))
        data.append([train["Summary"][i],str(train["Placed"][i])])
    word_features = get_word_features(get_words_in_sentences(sentences))
    print(data)
    def extract_features(document):
        document_words = set(document)
        features = {}
        for word in word_features:
          features['contains(%s)' % word] = (word in document_words)
        return features
    training_set = nltk.classify.util.apply_features(extract_features, data)
    time.sleep(5)
    classifier = nltk.classify.SklearnClassifier(LinearSVC())
    classifier.train(training_set)
##    classifier = nltk.NaiveBayesClassifier.train(training_set)
    f=open("myclass.pickle","wb")
    pickle.dump(classifier,f)
    f.close



mainn()


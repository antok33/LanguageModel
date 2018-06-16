# encoding: utf-8

import cPickle as pickle
import glob
import math
import time
from nltk import bigrams,re
from nltk import trigrams
from nltk import word_tokenize
from nltk import sent_tokenize
import operator
from random import randint


def parseEuroparl():
    txt = ""
    test_txt = ""
    cc = 0
    print "Reading Files ..."
    X = len(glob.glob('/home/antok/Downloads/txt/en/*.txt'))
    for i,filename in enumerate(glob.glob('/home/antok/Downloads/txt/en/*.txt')):
    	print "Progress %d of %d\r" %(i, X),
        if cc < 500:
            with open(filename, "r") as f:
                for line in f:
                    if line not in ['\n', '\r\n']:
                        line = re.sub('<.*.>', "", line)
                        txt += line
                if len(txt) != 0:
                    if txt[len(txt)-1] != ".":
                        txt += "."
            cc += 1
        elif cc >= 500 and cc < 650:
            with open(filename, "r") as f:
                for line in f:
                    if line not in ['\n', '\r\n']:
                        line = re.sub('<.*.>', "", line)
                        test_txt += line
                    if test_txt[len(test_txt)-1] != ".":
                        test_txt += "."
            cc += 1
        else:
            break

    print "decoding to utf-8 ..."
    txt = txt.decode('utf-8')
    test_txt = test_txt.decode('utf-8')
    
    print "tokenization ..."
    sentences = [sent for sent in sent_tokenize(txt)]
    test_sentences = [sent for sent in sent_tokenize(test_txt)]
    
    return txt, test_txt, sentences, test_sentences

def vocabularyGenerator(sentences):
    print "Building Vocabulary ..."
    wordsDict = {}
    
    for (i,sentence) in enumerate(sentences):
        # print "Progress %d of %d\r" %(i, len(sentences)),
        for w in word_tokenize(sentence):
            if w not in wordsDict:
                wordsDict[w] = 1
            else:
                wordsDict[w] += 1
    return wordsDict

def buildGramsVocabulary(unk_txt):
    print "Building Vocabulary  for 2-grams, 3-grams ..."
    bigramsDict = {}
    trigramsDict = {}
    sentences = [sent for sent in sent_tokenize(unk_txt)]
    
    for (i,sentence) in enumerate(sentences):
        for w1, w2 in bigrams(word_tokenize(sentence), pad_right=True, pad_left=True):
            if (w1,w2) not in bigramsDict:
                bigramsDict[(w1,w2)] = 1
            else:
                bigramsDict[(w1,w2)] += 1
        for w1, w2, w3 in trigrams(word_tokenize(sentence), pad_right=True, pad_left=True):
            if (w1,w2,w3) not in trigramsDict:
                trigramsDict[(w1,w2,w3)] = 1
            else:
                trigramsDict[(w1,w2,w3)] += 1
    
    pickle.dump(bigramsDict, open("bigramsDictUNK.p", "wb")) 
    pickle.dump(trigramsDict, open("trigramsDictUNK.p", "wb"))
    return bigramsDict, trigramsDict
    
    
def PLaplace_Bigram(wordsDict, bigrammsDict, w1, w2):
    V = len(wordsDict)
    
    if w1 in wordsDict:
        Cw1 = wordsDict[w1]
        if (w1, w2) in bigrammsDict:
            Cw1w2 = bigrammsDict[(w1,w2)]
        else:
            Cw1w2 = 0
    else:
        Cw1 = 0
        Cw1w2 = 0
    pl = (Cw1w2 + 1) / float(Cw1 + V)
    logpl = -math.log(pl,2)
    return pl, logpl

def PLaplace_Trigram(wordsDict, bigramsDict, trigramsDict, w1, w2, w3):
    V = len(wordsDict)
    flag = False
    if w1 not in wordsDict or w2 not in wordsDict:
        Cw1w2 = 0
        flag = True
    else:    
        if (w1,w2) in bigramsDict:
            Cw1w2 = bigramsDict[(w1,w2)]
        else:
            Cw1w2 = 0
    
    if flag or w3 not in wordsDict:
        Cw1w2w3 = 0
    else: 
        if (w1, w2, w3) in trigramsDict:
            Cw1w2w3 = trigramsDict[(w1,w2,w3)]
        else:
            Cw1w2w3 = 0
    pl = (Cw1w2w3 + 1) / float(Cw1w2 + V)
    logpl = -math.log(pl,2)
    return pl, logpl


def predict_Next_word(word, bigrammsDict):
    candidateWords = {}

    for w in bigrammsDict:
        if w[0] == word:
            candidateWords[w[1]] = bigrammsDict[(word, w[1])]

    sorted_words = sorted(candidateWords.items(), key=operator.itemgetter(1), reverse=True)
    for (i,word) in enumerate(sorted_words):
        if i > 2:
            break
        print word[0]

def sentencePropabilityBigrams(wordsDict, bigramsDict, sentence):
    print "Calculating Sentence Propability Bigram Model ..."
    Pr = 1
    for (w1, w2) in bigrams(word_tokenize(sentence), pad_right=True, pad_left=True):
        Pr *= PLaplace_Bigram(wordsDict, bigramsDict, w1, w2)[0]
        #Pr *= Kneser_Ney_Bigramm(wordsDict, bigramsDict, w1, w2) 
    print  "Sentence Propability for Bigram Model:", Pr
    return Pr

def sentenceLogPropabilityBigrams(wordsDict, bigramsDict, sentence):
    print "Calculating Sentence Log Propability Bigram Model ..."
    S = 0
    for (w1, w2) in bigrams(word_tokenize(sentence), pad_right=True, pad_left=True):
        S += PLaplace_Bigram(wordsDict, bigramsDict, w1, w2)[1]
        #Pr *= Kneser_Ney_Bigramm(wordsDict, bigramsDict, w1, w2) 
    print "Sentence Log Propability for Bigram Model:", S
    #print "Cross Entropy for Trigram model: ", S/float(len(word_tokenize(sentence)))
    return S, S/float(len(word_tokenize(sentence)))

def sentencePropabilityTrigrams(wordsDict, bigramsDict, trigramsDict, sentence):
    print "Calculating Sentence Propability Trigram Model ..."
    Pr = 1
    for (w1, w2, w3) in trigrams(word_tokenize(sentence), pad_right=True, pad_left=True):
        Pr *= PLaplace_Trigram(wordsDict, bigramsDict, trigramsDict, w1, w2, w3)[0]
        #Pr *= Kneser_Ney_Bigramm(wordsDict, bigramsDict, w1, w2) 
    print  "Sentence Propability for Trigram Model:", Pr
    return Pr

def sentenceLogPropabilityTrigrams(wordsDict, bigramsDict, trigramsDict, sentence):
    print "Calculating Sentence Log Propability Trigram Model..."
    S = 0
    for (w1, w2, w3) in trigrams(word_tokenize(sentence), pad_right=True, pad_left=True):
        S += PLaplace_Trigram(wordsDict, bigramsDict, trigramsDict, w1, w2, w3)[1]
    print "Sentence Log Propability for Trigram Model :", S
    #print "Cross Entropy for Trigram model: ", S/float(len(word_tokenize(sentence)))
    return S, S/float(len(word_tokenize(sentence)))

def testdataEvaluationBigram(wordsDict, bigramsDict, sentences):
    print "TEST SET EVALUATION (BIGRAM MODEL) ..."
    S = 0
    N = 0
    for (i,sentence) in enumerate(sentences):
        print "Progress %d of %d\r" %(i, len(sentences)),
        for (w1, w2) in bigrams(word_tokenize(sentence), pad_right=True, pad_left=True):
            S += PLaplace_Bigram(wordsDict, bigramsDict, w1, w2)[1]
        N += len(word_tokenize(sentence))
    print "Log propabilities for test set (bigram model): ", S
    print "Cross Entropy for test set (bigram model): ", S/float(N)     
    print "Perplexity for test set (bigram model): ", 2 ** (S/float(N))
    
def testdataEvaluationTrigram(wordsDict, bigramsDict, trigramsDict, sentences):
    print "TEST SET EVALUATION (TRIGRAM MODEL) ..."
    S = 0
    N = 0
    for (i,sentence) in enumerate(sentences):
        print "Progress %d of %d\r" %(i, len(sentences)),
        for (w1, w2, w3) in trigrams(word_tokenize(sentence), pad_right=True, pad_left=True):
            S += PLaplace_Trigram(wordsDict, bigramsDict, trigramsDict, w1, w2, w3)[1]
        N += len(word_tokenize(sentence))
    print "Log propabilities for test set (trigram model): ", S
    print "Cross Entropy for test set (trigram model): ", S/float(N)
    print "Perplexity for test set (trigram model): ", 2 ** (S/float(N))
        

if __name__ == '__main__':

    
    print "CREATING VOCABULARY       ---> PRESS 1"
    print "READ VOCABULARY FROM DISC ---> PRESS 2"
    voc = raw_input("1/2: ")
    if voc == "1":
        
        # GENERATE NEW V (uncomment the following three lines and comment the 'READ FROM DISK BLOCK')
        txt, test_txt, sentences, test_sentences = parseEuroparl()
        wordsDict = vocabularyGenerator(sentences)
        newwordsDict = {}
        unk = []
        print "Inserting *UNK* to test set ..."
        for (i,sent) in enumerate(sentences):
            print "Progress %d of %d\r" %(i, len(sentences)),
            for word in word_tokenize(sent):
                if wordsDict[word] > 4:
                    newwordsDict[word] = wordsDict[word]
                    unk.append(word)
                else:
                    newwordsDict["*UNK*"] = wordsDict[word]
                    unk.append("*UNK*")
        unk_txt = " ".join(unk)
       
        pickle.dump(newwordsDict, open("wordsDictUNK.p", "wb"))
        bigramsDict, trigramsDict = buildGramsVocabulary(unk_txt)
        
        print "Inserting *UNK* in test set ..."
        unk_test = [] 
        for i,sent in enumerate(test_sentences):
            print "Progress %d of %d\r" %(i, len(test_sentences)),
            for word in word_tokenize(sent):
                if word not in newwordsDict:
                    unk_test.append("*UNK*")
                else:
                    unk_test.append(word)
        unk_test_txt = " ".join(unk_test)
                    
        print "tokenize test set sentences with *UNK* ..."
        test_sentences = [sent for sent in sent_tokenize(unk_test_txt)]         
        
        pickle.dump(test_sentences, open("test_sentences.p", "wb"))
        
    elif voc == "2":
        # READ V FROM DISK
        test_sentences = pickle.load(open("test_sentences.p", "rb" ))
        print "Reading wordsDict ..."
        wordsDict = pickle.load(open("wordsDictUNK.p", "rb" ))
        print "Reading bigramsDict ..."
        bigramsDict = pickle.load(open("bigramsDictUNK.p", "rb" ))
        print "Reading trigramsDict ..."
        trigramsDict = pickle.load(open("trigramsDictUNK.p", "rb" ))                
    
    
    choice = ""
    while choice != "exit":
        print ""
        print "COMPARE TEST SENTENCE VS RANDOM SENTENCE ---> PRESS 1"
        print "FOR NEXT WORD PREDICTION                 ---> PRESS 2"
        print "FOR TEST SET EVALUATION                  ---> PRESS 3"
        print ""
        choice = raw_input("1/2/3 (Press exit to stop execution): ")
        if choice == "1":
            
            # PICK A RANDOM SENTENCE FROM THE TEST SET
            sent_id = randint(0, len(test_sentences))
            print len(test_sentences)
            test_sentence = test_sentences[sent_id]
            print "Test data Sentence:\n", test_sentence, "\n"
            
            # GENERATE RANDOM SENTENCE FROM VOCABULARY
            random_sentence = ""
            while len(word_tokenize(random_sentence)) <= len(word_tokenize(test_sentence)) -2:
                word_id = randint(0, len(wordsDict))
                random_sentence += wordsDict.keys()[word_id] + " "    
            random_sentence += "."
            print "Random sentence: \n", random_sentence, "\n"
            
            print "LOG PROPABILITY FOR TEST SENTENCE (BIGRAM MODEL)"
            sentenceLogPropabilityBigrams(wordsDict, bigramsDict, test_sentence)
            print "LOG PROPABILITY FOR TEST SENTENCE (TRIGRAM MODEL)"
            sentenceLogPropabilityTrigrams(wordsDict, bigramsDict, trigramsDict, test_sentence)
            
            print ""
            
            print "LOG PROPABILITY FOR RANDOM SENTENCE (BIGRAM MODEL)"
            sentenceLogPropabilityBigrams(wordsDict, bigramsDict, random_sentence)
            print "LOG PROPABILITY FOR RANDOM SENTENCE (TRIGRAM MODEL)"
            sentenceLogPropabilityTrigrams(wordsDict, bigramsDict, trigramsDict, random_sentence)
        
        elif choice == "2":
            w1 = ""
            while w1 != "*EXIT*":
                w1 = raw_input("Enter a word (Enter *EXIT* to stop prediction): ")
                # PREDICTION
                print("Prediction: ")
                predict_Next_word(w1, bigramsDict)
        
        elif choice == "3":
            testdataEvaluationBigram(wordsDict, bigramsDict, test_sentences) 
            testdataEvaluationTrigram(wordsDict, bigramsDict, trigramsDict, test_sentences)


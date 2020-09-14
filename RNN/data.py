import torch
from torch.utils.data import TensorDataset, DataLoader
import re
import os
os.system('pip install nltk')
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from collections import Counter
import numpy as np
import pandas as pd


class Dataset(object):
    def __init__(self, path=""):
        self.path = path
        
    def getDictionaryFromS3(self, fileName):
        dictPath = os.path.join(self.path, fileName)        
        with open(dictPath, 'rb') as f:                                
            dictionary = torch.load(f)
        return dictionary
        
    def getDatasetTrain(self, batch_size):
        trainDataset = pd.read_csv(os.path.join(self.path, "train"), header=None, names=None)
        
        trainY = torch.from_numpy(trainDataset[[0]].values).float().squeeze()
        trainX = torch.from_numpy(trainDataset.drop([0], axis=1).values).long()
        print("Train dataset len = {}".format(len(trainDataset)))
        traindata = TensorDataset(trainX, trainY)
        print("Tensor train dataset {}".format(traindata))
        return DataLoader(traindata, shuffle=True, batch_size=batch_size, drop_last=True)
        
    def getDatasetTest(self, batch_size):
        testDataset = pd.read_csv(os.path.join(self.path, "test"), header=None, names=None)
        
        testY = torch.from_numpy(testDataset[[0]].values).float().squeeze()
        testX = torch.from_numpy(testDataset.drop([0], axis=1).values).long()
        testdata = TensorDataset(testX, testY)
        result = DataLoader(testdata, shuffle=True, batch_size=batch_size, drop_last=True)
        
        return result
        
     
    def getDataValid(self, batch_size):
        validDataset = pd.read_csv(os.path.join(self.path, "valid"), header=None, names=None)
        
        validY = torch.from_numpy(validDataset[[0]].values).float().squeeze()
        validX = torch.from_numpy(validDataset.drop([0], axis=1).values).long()
        
        validdata = TensorDataset(validX, validY)
        
        return DataLoader(validdata, shuffle=True, batch_size=batch_size, drop_last=True)
    
    def transformRawData(self, words2index, rawData, seq_len):
        clean = re.compile(r'<.*?>')

        print(rawData)

        cleanr = re.compile(r"[^a-zA-Z0-9]")
        stop_words = set(stopwords.words('english')) 

        filtered_sentence = []

        for text in rawData:
            notTagData = re.sub(clean, '', text)
            word_tokens = word_tokenize(re.sub(cleanr, ' ', notTagData.lower())) # All words in lower case
            filtered_sentence.append([w for w in word_tokens if not w in stop_words])   

        print(filtered_sentence)

        tokenizedWord = [[]]

        for i, sentence in enumerate(filtered_sentence):
            sentence = re.sub("[^a-zA-Z]",  " ", str(sentence))
            tokenizedWord[i] = [words2index[word] if word in words2index else words2index['UNKN_'] for word in 
                         nltk.word_tokenize(sentence)]

        print(tokenizedWord)

        redefinedText = np.zeros((len(tokenizedWord), seq_len),dtype=int)
        for ii, review in enumerate(tokenizedWord):
            if len(review) != 0:
                redefinedText[ii, -len(review):] = np.array(review)[:seq_len]

        print(redefinedText)
        
        result = DataLoader(redefinedText, shuffle=True, batch_size=seq_len)
        
        print(result)
        
        return result



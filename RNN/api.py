import argparse
import json
import os
import pickle
import joblib
import sys
import sagemaker_containers
import pandas as pd
import torch as torch
from RNN import IMDBClassifier
from data import Dataset
import numpy as np


def model_fn(model_dir):
    print("Initializing model_fn.")
        
    modelCfg = {}    
    model_cfg = os.path.join(model_dir, 'model.cfg')
    with open(model_cfg, 'rb') as f:                                
        modelCfg = torch.load(f)  
    
    completeModel = IMDBClassifier(modelCfg['vocab_size'], modelCfg['output_size'], modelCfg['embedding_dim'], modelCfg['hidden_dim'], modelCfg['n_layers'])    

    complete_model = os.path.join(model_dir, 'model.pth')
    with open(complete_model, 'rb') as f:
        completeModel.load_state_dict(torch.load(f))

    complete_dict = os.path.join(model_dir, 'model.dic')
    with open(complete_dict, 'rb') as f:
        completeModelDict = torch.load(f)

    completeModel.to(torch.device(modelCfg['device']))
    completeModel.dictionary = completeModelDict

    print(type(completeModel))
    print('____')
    print(completeModel)
    print('____')
    
    return completeModel

def input_fn(serialized_input_data, content_type):
    print('Deserializing the input data.')
    print(serialized_input_data)
    text = ""
    if content_type == 'application/json':
        obj = json.loads(serialized_input_data)
        print('Predicting text='+obj['text'])
        
        text = str(obj['text'])
    elif content_type == 'text/plain':
        text = serialized_input_data.decode('utf-8')    
        print('Predicting text='+text)
    else:
        raise Exception('Requested unsupported ContentType in content_type: ' + content_type)
        
    return text    
    

def output_fn(prediction_output, accept):
    print('Serializing the generated output.')
    
    if accept == "application/json":        
        return json.JSONEncoder().encode({"algorithm":"RNN","answer": str(prediction_output)})
    elif content_type == 'text/plain':
        return str(prediction_output)
    else:
        raise Exception('Requested unsupported ContentType in content_type: ' + content_type)    
    

def predict_fn(input_data, model):
    print('Inferring from transformed input data.')
    
    dataset = Dataset('')
    
    x = dataset.transformRawData(model.dictionary, [input_data], model.seq_len)   
    
    h = model.init_hidden(1)
    
    h = tuple([each.data for each in h])   

    inp = next(iter(x))

    output, h = model(inp, h)
    
    prediction = int(np.round(output.detach().numpy()))
    
    print(" ### H {}".format(h))
    print(" ### Output {}".format(output))
    print(" ### Result {}".format(prediction))   
    
    return prediction
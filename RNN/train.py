import os
import argparse
import json
import pickle
import sys
import sagemaker_containers
import pandas as pd
import torch as torch
import numpy as np
from RNN import IMDBClassifier
from data import Dataset

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

def train(model_dir, model, train_loader, epochs, optimizer, loss_fn, device, val_loader, batch_size, clip):
    counter = 0
    print_every = 100    
    
    valid_loss_min = np.Inf
    
    model.train()
    print("Initializing training.")
    for i in range(epochs):
        h = model.init_hidden(batch_size)        
        for inputs, labels in train_loader:
            counter = counter + 1
            h = tuple([e.data for e in h])
            inputs, labels = inputs.to(device), labels.to(device)
            model.zero_grad()
            
            output, h = model(inputs, h)
            loss = criterion(output.squeeze(), labels.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            if counter%print_every == 0 or counter == len(train_loader)-1:
                val_h = model.init_hidden(batch_size)
                val_losses = []
                model.eval()
                for inp, lab in val_loader:
                    val_h = tuple([each.data for each in val_h])
                    inp, lab = inp.to(device), lab.to(device)
                    out, val_h = model(inp, val_h)
                    val_loss = criterion(out.squeeze(), lab.float())
                    val_losses.append(val_loss.item())

                model.train()
                print("Epoch: {}/{}...".format(i+1, epochs),
                      "Step: {}...".format(counter),
                      "Loss: {:.6f}...".format(loss.item()),
                      "Val Loss: {:.6f}".format(np.mean(val_losses)))
                if np.mean(val_losses) <= valid_loss_min: 
                    with open(os.path.join(model_dir, 'model.pth'), 'wb') as f:
                        torch.save(model.state_dict(), f)                    
                        
                    print('Validation loss decreased ({:.6f} --> {:.6f}).Saving model...'.format(valid_loss_min,np.mean(val_losses)))
                    valid_loss_min = np.mean(val_losses)

                    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Training and model Parameters
    parser.add_argument('--batch_size', type=int, default=400, metavar='N')
    parser.add_argument('--epochs', type=int, default=10, metavar='N')
    parser.add_argument('--seed', type=int, default=1, metavar='S')    
    parser.add_argument('--embedding_dim', type=int, default=400, metavar='N')
    parser.add_argument('--hidden_dim', type=int, default=512, metavar='N')
    parser.add_argument('--n_layers', type=int, default=2, metavar='N')
    parser.add_argument('--vocab_size', type=int, default=5000, metavar='N')
    parser.add_argument('--output_size', type=int, default=1, metavar='N')
    parser.add_argument('--seq_len', type=int, default=200, metavar='N')
    parser.add_argument('--drop_prob', type=float, default=0.5, metavar='N')
    parser.add_argument('--dropout', type=float, default=0.2, metavar='N')
    parser.add_argument('--clip', type=int, default=5, metavar='N')        
    parser.add_argument('--dictionary_file_name', type=str, default='dictionary', metavar='N')

    # SageMaker Parameters
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])

    args = parser.parse_args()
    
    is_cuda = torch.cuda.is_available()
    
    device = ""
    deviceStr = "cpu"
    
    if is_cuda:
        device = torch.device("cuda")
        deviceStr = "cuda"
        print("GPU is available")
    else:
        device = torch.device("cpu")
        print("GPU not available, CPU used")
    
    print("Using device {}.".format(device))

    torch.manual_seed(args.seed)
    
    dataset = Dataset(args.data_dir)
    
    train_loader = dataset.getDatasetTrain(args.batch_size)
    test_loader = dataset.getDatasetTest(args.batch_size)
    
    val_loader = dataset.getDataValid(args.batch_size)    
    dictionary = dataset.getDictionaryFromS3(args.dictionary_file_name)    
    
    modelConfig = { 
                "embedding_dim" : args.embedding_dim,
                "hidden_dim"    : args.hidden_dim,
                "vocab_size"    : args.vocab_size,
                "output_size"   : args.output_size,
                "n_layers"      : args.n_layers,
                "device"        : deviceStr,
                "batch_size"    : args.batch_size,
        }
    
    model_cfg = os.path.join(args.model_dir, 'model.cfg')
    with open(model_cfg, 'wb') as f:                                
        torch.save(modelConfig, f)     
    
    with open(os.path.join(args.model_dir, 'model.dic'), 'wb') as f:
        torch.save(dictionary, f)
    
    model = IMDBClassifier(args.vocab_size, args.output_size, args.embedding_dim, args.hidden_dim, args.n_layers)
    model.to(device)
    model.dictionary = dictionary
    model.batch_size = args.batch_size
    model.seq_len = args.seq_len
    
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)            
    
    
    train(args.model_dir, model, train_loader, args.epochs, optimizer, criterion, device, val_loader, args.batch_size, args.clip)   
    
    print("Loading best model...")
    # Load best model    
    model_path = os.path.join(args.model_dir, 'model.pth')
    
    with open(model_path, 'rb') as f:                
        model.load_state_dict(torch.load(f))     
    
        
    test_losses = []
    num_correct = 0
    h = model.init_hidden(args.batch_size)
    model.eval()
    for inputs, labels in test_loader:
        try:
            if inputs == None or labels == None:
                print("problem inputs {} labels{}".format(inputs, labels))
                pass
            h = tuple([each.data for each in h])
            inputs, labels = inputs.to(device), labels.to(device)
            print("Predicting...")
            print("  inputs len = {}".format(len(inputs)))
            print("  h len = {}".format(len(h)))
            output, h = model(inputs, h)
            print("Result = {}".format(output))
            test_loss = criterion(output.squeeze(), labels.float())
            test_losses.append(test_loss.item())
            pred = torch.round(output.squeeze()) 
            correct_tensor = pred.eq(labels.float().view_as(pred))
            correct = np.squeeze(correct_tensor.cpu().numpy())
            num_correct += np.sum(correct)
        except:
            pass
            

    print("Test loss: {:.3f}".format(np.mean(test_losses)))
    test_acc = num_correct/len(test_loader.dataset)
    print("Test accuracy: {:.3f}%".format(test_acc*100))
    
    
import torch
import torch.nn as nn

class IMDBClassifier(nn.Module):
    def __init__(self, vocab_size=5000, output_size=1, embedding_dim=400, hidden_dim=512, n_layers=2, drop_prob=0.5, dropout=0.2):
        super(IMDBClassifier, self).__init__()
        self.dictionary = None
        self.batch_size = 400
        self.seq_len = 200
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_features=hidden_dim, out_features=output_size)
        self.sigmoid = nn.Sigmoid()          
        
    def forward(self, x, hidden):
        batch_size = x.size(0)
        x = x.long()
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        
        out = self.dropout(lstm_out)
        out = self.fc(out)
        out = self.sigmoid(out)
        
        out = out.view(batch_size, -1)
        out = out[:,-1]
        return out, hidden    
        
    
    def init_hidden(self, batch_size):
        is_cuda = torch.cuda.is_available()

        if is_cuda:
            device = torch.device("cuda")            
        else:
            device = torch.device("cpu")
            
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))
        return hidden
"""
Basic LSTM Class with bi-directional option
Authors:                    Arnold Yeung
Upload Date:                April 6th, 2019
Revisions:
    2019-04-06      (AY)    Initial upload
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class lstm_rnn(nn.Module):
    
    def __init__(self, params):
        super().__init__()
        
        self.input_dim = params['input_dim']
        self.hidden_dim = params['hidden_dim']
        self.output_dim = params['output_dim']
        self.num_layers = params['num_layers']
        self.bidirectional = params['bidirectional']
        
        self.batch_size = params['batch_size']
        self.hidden_states = self.init_hidden_states(params['bidirectional'])
        self.cell_state = self.init_cell_state(params['bidirectional'])
        
        #   LSTM hidden layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, 
                            bidirectional=self.bidirectional)
        self.linear = nn.Linear(self.hidden_dim, self.output_dim)
        self.sigmoid = nn.Sigmoid()
    
    def init_hidden_states(self, bidirection=False):
        """
        Initialize hidden states to random.
        Arguments:
            bidirection (boolean) : whether LSTM is bidirectional or not
        Returns:
            hidden state (torch.tensor) : hidden states per layer (1 state per layer)
        """
        if bidirection is True:
             return torch.randn(self.num_layers*2, self.batch_size, self.hidden_dim)
        else:
            return torch.randn(self.num_layers, self.batch_size, self.hidden_dim)
    
    def init_cell_state(self, bidirection=False):
        """
        Initialize cell states to 0.
        Arguments:
            bidirection (boolean) : whether LSTM is bidirectional or not
        Returns:
            cell state (torch.array) : cell states per layer (1 state per layer)
        """
        if bidirection is True:
             return torch.zeros(self.num_layers*2, self.batch_size, self.hidden_dim)
        else:
            return torch.zeros(self.num_layers, self.batch_size, self.hidden_dim)
        
    def forward(self, X):
        """
        Forward pass of model
        Arguments:
            input:    input matrix of size(seq_length, batch_size, input_dim)
        Return:
            y_pred:   predicted output of each sequence step
        """
        
        #   lstm_out is output for each t in sequence length T
        #   final_hidden_states are hidden_states for t=T
        #   final_cell_state is cell_state for t=T
        
        lstm_out, (final_hidden_states, final_cell_state) = self.lstm(X, 
                  (self.hidden_states, self.cell_state))
        
        #   convert lstm_out back to tensor from packedsequence
        lstm_out = nn.utils.rnn.pad_packed_sequence(lstm_out)

        y_pred = torch.sigmoid(self.linear(final_hidden_states[-1]))
        
        return y_pred
        
    def compute_loss(self, y, target):
        """
        Compute the loss function using the MSE Loss function
        Arguments:
            y (torch.tensor) : outputs of classifier for input (batch_size, output_dim)
            target (torch.tensor) : ground truth labels for input (batch_size, output_dim)
        Returns:
            loss (torch.tensor) : MSE loss of y
        """
        return F.mse_loss(y, target)
    
def train(X_train, y_train, clf, params):
    """
    Trains the model given training data.
    Arguments:
        X_train (PackedSequence) : packed (tensor) sequences of size seq_len x 
            num_sample x dimensions
        y_train (torch.tensor()) : expected output labels corresponding to X
        clf (class) : lstm model
        params (dict) : contains model, dimension, and batch parameters
    Returns:
        None
    """
    
    num_epochs = params['num_epochs']
    lr = params['learning_rate']
    #   Create optimizer
    
    optimizer = torch.optim.Adam(list(clf.parameters()), lr=lr)

    for epoch in range(0, num_epochs):
        
        #   get batch
        X_train_batch, y_train_batch = X_train, y_train
        
        #   Zero the gradient of the optimizer
        optimizer.zero_grad()
        
        #   Forward pass
        y_pred_batch = clf.forward(X_train_batch)
        loss = clf.compute_loss(y_pred_batch, y_train_batch)
        
        print("Epoch: " + str(epoch) + "    Loss: " + str(loss.item()))
    
        loss.backward()     #   backward pass
        optimizer.step()    #   update gradient step

def evaluate(X_test, y_test, clf, params):
    
    y_pred = clf.forward(X_test)
    loss = clf.compute_loss(y_pred, y_test)
    
    return loss, y_pred

def create_sample_data(params, min_seq_len, max_seq_len):
    """
    Creates data X, y for a very simple test case.  For first half of the batch
    will be labelled 1 and will have X < 1.  The second half of the batch will
    be labelled 0 and will have X typically > 1.
    Arguments:
        params (dict) : contains dimension and batch parameters
        min_seq_len (int) : length of the shortest possible sequence
        max_seq_len (int) : length of the longest possible sequence
    Returns:
        Packed padded sequence containing all sequences in X
        torch.tensor(batch_size, output_dim) : y labels corresponding to X
    """
    
    input_dim = params['input_dim']
    output_dim = params['output_dim']
    batch_size = params['batch_size']
    
    X = []
    
    #   generate random sorted seq_len
    seq_len = torch.randint(min_seq_len, max_seq_len, (batch_size,))       
    seq_len = seq_len.sort(descending=True)[0]
    
    #   create y
    y = torch.ones((batch_size,output_dim))
    y[int(batch_size/2):, ] = 0
    
    #   create data sequences of different sequence lengths
    for i in range(0, batch_size):
        sequence = torch.randn((seq_len[i], input_dim))
        if i > int(batch_size/2)-1:    #   create difference in later half of X
            sequence *= 50
        
        X.append(sequence)

    X = nn.utils.rnn.pad_sequence(X)
    X = nn.utils.rnn.pack_padded_sequence(X, seq_len)

    return X, y

def main_fcn(params, verbose=False):
    
    #   initialize classifier
    if verbose:
        print("Initializing classifier...")
        
    #   Create model
    clf = lstm_rnn(params)
    
    #   Train model
    X_train, y_train = create_sample_data(params, 2, 12)
    X_test, y_test = create_sample_data(params, 2, 12)
    train(X_train, y_train, clf, params)
    
    #   Evaluate model
    loss, output = evaluate(X_test, y_test, clf, params)
    print(loss)
    print(output)
    
if __name__ == '__main__':
    
    torch.manual_seed = 1
    vocab_size = 3
    
    params = {
            'input_dim':        vocab_size,
            'hidden_dim':       20,            
            'output_dim':       1,              
            'num_layers':       5,
            'batch_size':       2,
            'num_epochs':       200,
            'learning_rate':    0.05,
            'learning_decay':   0.9,
            'bidirectional':    True
    }
    
    main_fcn(params, verbose=True)
